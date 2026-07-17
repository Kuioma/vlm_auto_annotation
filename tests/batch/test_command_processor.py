import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from auto_annotation.batch.models import BatchItem
from auto_annotation.batch.processors.command import (
    PLUGIN,
    CommandProcessorConfig,
    CommandProcessorError,
    CommandProcessorProvider,
)


def _item(tmp_path: Path, *, item_id: str = "episode-1") -> BatchItem:
    head = tmp_path / "head view.mp4"
    head.write_bytes(b"head")
    right = tmp_path / "right view.mp4"
    right.write_bytes(b"right")
    return BatchItem(
        item_id=item_id,
        dataset_id="wa2-demo",
        media={"HEAD_RGB": head, "RIGHT_WRIST_RGB": right},
        primary_media="HEAD_RGB",
        context={"task": "pick and place the rubber"},
        metadata={"episode_index": 1, "length": 236},
        input_fingerprint=f"sha256:{'a' * 64}",
    )


def _context(tmp_path: Path) -> SimpleNamespace:
    workspace = tmp_path / "workspace"
    attempt_dir = workspace / "attempt-1"
    output_dir = attempt_dir / "output"
    output_dir.mkdir(parents=True)
    return SimpleNamespace(
        run_id="run-test",
        workspace=workspace,
        attempt_dir=attempt_dir,
        output_dir=output_dir,
        attempt=1,
    )


def _run_processor(
    provider: CommandProcessorProvider,
    item: BatchItem,
    context: SimpleNamespace,
):
    async def execute():
        async with provider.open() as session:
            return await session.process(item, context)

    return asyncio.run(execute())


def test_command_uses_shell_false_and_expands_supported_templates(
    tmp_path: Path,
) -> None:
    item_id = "episode;touch should-not-exist"
    item = _item(tmp_path, item_id=item_id)
    context = _context(tmp_path)
    script = (
        "import json, pathlib, sys; "
        "out = pathlib.Path(sys.argv[4]); "
        "(out / 'argv.json').write_text(json.dumps(sys.argv[1:])); "
        "(out / 'output.jsonl').write_text("
        "json.dumps({{'item_id': sys.argv[1]}}) + '\\n'); "
        "print('stdout-only'); "
        "print('stderr-only', file=sys.stderr)"
    )
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(
                sys.executable,
                "-c",
                script,
                "{item_id}",
                "{input_path}",
                "{media_HEAD_RGB}",
                "{output_dir}",
                "{workspace}",
            ),
        )
    )

    result = _run_processor(provider, item, context)

    assert result.record == {"item_id": item_id}
    assert json.loads(
        (context.output_dir / "argv.json").read_text(encoding="utf-8")
    ) == [
        item_id,
        str(item.input_path),
        str(item.media["HEAD_RGB"]),
        str(context.output_dir),
        str(context.workspace),
    ]
    assert not (Path.cwd() / "should-not-exist").exists()
    assert (context.attempt_dir / "stdout.log").read_text(
        encoding="utf-8"
    ) == "stdout-only\n"
    assert (context.attempt_dir / "stderr.log").read_text(
        encoding="utf-8"
    ) == "stderr-only\n"
    assert result.artifacts == tuple(
        sorted(
            (
                context.output_dir / "argv.json",
                context.output_dir / "output.jsonl",
            ),
            key=lambda path: path.as_posix(),
        )
    )


def test_writes_canonical_item_snapshot_and_expands_item_json_template(
    tmp_path: Path,
) -> None:
    item = _item(tmp_path)
    context = _context(tmp_path)
    script = (
        "import json, pathlib, sys; "
        "item = json.loads(pathlib.Path(sys.argv[1]).read_text()); "
        "out = pathlib.Path(sys.argv[2]); "
        "(out / 'output.jsonl').write_text("
        "json.dumps({{'item_id': item['item_id']}}) + '\\n')"
    )
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(
                sys.executable,
                "-c",
                script,
                "{item_json}",
                "{output_dir}",
            ),
        )
    )

    result = _run_processor(provider, item, context)

    item_json = context.attempt_dir / "item.json"
    expected_snapshot = {
        "item_id": item.item_id,
        "dataset_id": item.dataset_id,
        "media": {
            alias: str(path) for alias, path in item.media.items()
        },
        "primary_media": item.primary_media,
        "context": item.context,
        "metadata": item.metadata,
        "input_fingerprint": item.input_fingerprint,
    }
    assert result.record == {"item_id": item.item_id}
    assert item_json.read_bytes() == json.dumps(
        expected_snapshot,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def test_preflight_validates_media_templates_without_writing_files(
    tmp_path: Path,
) -> None:
    item = _item(tmp_path)
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=("annotate", "{media_HEAD_RGB}"),
            cwd="{media_RIGHT_WRIST_RGB}",
            environment={"ITEM_VIEW": "{media_RIGHT_WRIST_RGB}"},
        )
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    provider.preflight((item,))

    assert sorted(
        path.relative_to(tmp_path) for path in tmp_path.rglob("*")
    ) == before


def test_preflight_rejects_missing_media_alias_before_attempt(
    tmp_path: Path,
) -> None:
    item = _item(tmp_path)
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=("annotate", "{media_LEFT_WRIST_RGB}"),
            environment={"AUX_VIEW": "{media_DEPTH}"},
        )
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    with pytest.raises(CommandProcessorError) as captured:
        provider.preflight((item,))

    assert captured.value.kind == "template"
    assert captured.value.retryable is False
    assert "episode-1" in str(captured.value)
    assert "LEFT_WRIST_RGB" in str(captured.value)
    assert "DEPTH" in str(captured.value)
    assert sorted(
        path.relative_to(tmp_path) for path in tmp_path.rglob("*")
    ) == before


@pytest.mark.parametrize(
    ("result_format", "result_path", "payload"),
    (
        ("single_jsonl", Path("output.jsonl"), '{"ok":true}\n'),
        ("json", Path("record.json"), '{"ok":true}\n'),
    ),
)
def test_parses_strict_object_result_from_configured_relative_path(
    tmp_path: Path,
    result_format: str,
    result_path: Path,
    payload: str,
) -> None:
    item = _item(tmp_path)
    context = _context(tmp_path)
    script = (
        "import pathlib, sys; "
        "pathlib.Path(sys.argv[1]).write_text(sys.argv[2])"
    )
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(
                sys.executable,
                "-c",
                script,
                "{output_dir}/" + result_path.as_posix(),
                payload.replace("{", "{{").replace("}", "}}"),
            ),
            result_format=result_format,
            result_path=result_path,
        )
    )

    result = _run_processor(provider, item, context)

    assert result.record == {"ok": True}


def test_result_path_none_parses_stdout_without_mixing_stderr(
    tmp_path: Path,
) -> None:
    item = _item(tmp_path)
    context = _context(tmp_path)
    script = (
        "import sys; "
        "print('{{\"ok\": true}}'); "
        "print('diagnostic', file=sys.stderr)"
    )
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(sys.executable, "-c", script),
            result_path=None,
        )
    )

    result = _run_processor(provider, item, context)

    assert result.record == {"ok": True}
    assert (context.attempt_dir / "stderr.log").read_text(
        encoding="utf-8"
    ) == "diagnostic\n"


@pytest.mark.parametrize(
    "payload",
    (
        '{"one":1}\n{"two":2}\n',
        "[]\n",
        '{"duplicate":1,"duplicate":2}\n',
        '{"not_finite":NaN}\n',
    ),
)
def test_rejects_non_single_or_non_strict_jsonl_results(
    tmp_path: Path,
    payload: str,
) -> None:
    item = _item(tmp_path)
    context = _context(tmp_path)
    script = (
        "import pathlib, sys; "
        "(pathlib.Path(sys.argv[1]) / 'output.jsonl').write_text(sys.argv[2])"
    )
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(
                sys.executable,
                "-c",
                script,
                "{output_dir}",
                payload.replace("{", "{{").replace("}", "}}"),
            )
        )
    )

    with pytest.raises(CommandProcessorError) as captured:
        _run_processor(provider, item, context)

    assert captured.value.kind == "invalid_result"
    assert captured.value.retryable is False


def test_timeout_is_a_typed_retryable_error_and_preserves_logs(
    tmp_path: Path,
) -> None:
    item = _item(tmp_path)
    context = _context(tmp_path)
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(
                sys.executable,
                "-c",
                (
                    "import sys, time; "
                    "print('before-timeout', flush=True); "
                    "print('timeout-error', file=sys.stderr, flush=True); "
                    "time.sleep(1)"
                ),
            ),
            timeout_s=0.05,
        )
    )

    with pytest.raises(CommandProcessorError) as captured:
        _run_processor(provider, item, context)

    error = captured.value
    assert error.kind == "timeout"
    assert error.retryable is True
    assert error.returncode is None
    assert (context.attempt_dir / "stdout.log").read_bytes().startswith(
        b"before-timeout"
    )
    assert (context.attempt_dir / "stderr.log").read_bytes().startswith(
        b"timeout-error"
    )


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_timeout_terminates_descendant_process_group(
    tmp_path: Path,
) -> None:
    item = _item(tmp_path)
    context = _context(tmp_path)
    ready = tmp_path / "grandchild-ready"
    orphan_marker = tmp_path / "grandchild-survived"
    child_script = (
        "import pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text('ready')\n"
        "time.sleep(2)\n"
        "pathlib.Path(sys.argv[2]).write_text('orphan')\n"
    )
    parent_script = (
        "import pathlib, subprocess, sys, time\n"
        "ready = pathlib.Path(sys.argv[2])\n"
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], "
        "sys.argv[2], sys.argv[3]])\n"
        "deadline = time.monotonic() + 5\n"
        "while not ready.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "if not ready.exists():\n"
        "    raise RuntimeError('grandchild did not start')\n"
        "print('parent-before-timeout', flush=True)\n"
        "print('parent-timeout-error', file=sys.stderr, flush=True)\n"
        "time.sleep(10)\n"
    )
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(
                sys.executable,
                "-c",
                parent_script,
                child_script,
                str(ready),
                str(orphan_marker),
            ),
            timeout_s=1,
        )
    )

    with pytest.raises(CommandProcessorError) as captured:
        _run_processor(provider, item, context)

    assert captured.value.kind == "timeout"
    assert ready.is_file()
    assert (context.attempt_dir / "stdout.log").read_text(
        encoding="utf-8"
    ) == "parent-before-timeout\n"
    assert (context.attempt_dir / "stderr.log").read_text(
        encoding="utf-8"
    ) == "parent-timeout-error\n"
    time.sleep(1.25)
    assert not orphan_marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_cancellation_terminates_descendant_process_group_and_preserves_logs(
    tmp_path: Path,
) -> None:
    item = _item(tmp_path)
    context = _context(tmp_path)
    grandchild_ready = tmp_path / "grandchild-ready-for-cancel"
    parent_ready = tmp_path / "parent-ready-for-cancel"
    orphan_marker = tmp_path / "grandchild-survived-cancel"
    child_script = (
        "import pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text('ready')\n"
        "time.sleep(1)\n"
        "pathlib.Path(sys.argv[2]).write_text('orphan')\n"
        "time.sleep(2)\n"
    )
    parent_script = (
        "import pathlib, subprocess, sys, time\n"
        "grandchild_ready = pathlib.Path(sys.argv[2])\n"
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], "
        "sys.argv[2], sys.argv[4]])\n"
        "deadline = time.monotonic() + 5\n"
        "while (not grandchild_ready.exists() "
        "and time.monotonic() < deadline):\n"
        "    time.sleep(0.01)\n"
        "if not grandchild_ready.exists():\n"
        "    raise RuntimeError('grandchild did not start')\n"
        "print('parent-before-cancel', flush=True)\n"
        "print('parent-cancel-error', file=sys.stderr, flush=True)\n"
        "pathlib.Path(sys.argv[3]).write_text('ready')\n"
        "time.sleep(3)\n"
    )
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(
                sys.executable,
                "-c",
                parent_script,
                child_script,
                str(grandchild_ready),
                str(parent_ready),
                str(orphan_marker),
            ),
            timeout_s=10,
        )
    )

    async def cancel_after_ready() -> tuple[float, bool]:
        async with provider.open() as session:
            task = asyncio.create_task(session.process(item, context))
            deadline = asyncio.get_running_loop().time() + 5
            while not parent_ready.exists():
                if asyncio.get_running_loop().time() >= deadline:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    raise AssertionError("parent process did not become ready")
                await asyncio.sleep(0.01)

            cancel_started = asyncio.get_running_loop().time()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            cancel_elapsed = (
                asyncio.get_running_loop().time() - cancel_started
            )

            # This is longer than the grandchild's marker delay. If cancellation
            # only abandons the subprocess wait, the marker will be observable.
            await asyncio.sleep(1.25)
            return cancel_elapsed, orphan_marker.exists()

    cancel_elapsed, orphan_survived = asyncio.run(cancel_after_ready())

    stdout_path = context.attempt_dir / "stdout.log"
    stderr_path = context.attempt_dir / "stderr.log"
    assert cancel_elapsed < 1.0
    assert (
        orphan_survived,
        (
            stdout_path.read_text(encoding="utf-8")
            if stdout_path.exists()
            else None
        ),
        (
            stderr_path.read_text(encoding="utf-8")
            if stderr_path.exists()
            else None
        ),
    ) == (
        False,
        "parent-before-cancel\n",
        "parent-cancel-error\n",
    )


@pytest.mark.parametrize(
    ("returncode", "retryable"),
    ((75, True), (2, False)),
)
def test_exit_code_error_retryability_is_configured(
    tmp_path: Path,
    returncode: int,
    retryable: bool,
) -> None:
    item = _item(tmp_path)
    context = _context(tmp_path)
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "print('separate-error', file=sys.stderr); "
                    f"raise SystemExit({returncode})"
                ),
            ),
            retryable_exit_codes=(75,),
        )
    )

    with pytest.raises(CommandProcessorError) as captured:
        _run_processor(provider, item, context)

    error = captured.value
    assert error.kind == "exit_code"
    assert error.returncode == returncode
    assert error.retryable is retryable
    assert (context.attempt_dir / "stderr.log").read_text(
        encoding="utf-8"
    ) == "separate-error\n"


def test_identity_hashes_configured_files(tmp_path: Path) -> None:
    first = tmp_path / "processor.py"
    first.write_bytes(b"processor-v1")
    second = tmp_path / "contract.yaml"
    second.write_bytes(b"contract-v1")
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="processor-contract-v1",
            argv=(sys.executable, "--version"),
            identity_files=(first, second),
            cwd=tmp_path,
            environment={"PROCESSOR_MODE": "strict"},
        )
    )

    semantic_payload = {
        "argv": [sys.executable, "--version"],
        "cwd": str(tmp_path.resolve()),
        "environment": {"PROCESSOR_MODE": "strict"},
        "result_format": "single_jsonl",
        "result_path": "output.jsonl",
        "semantic_version": "processor-contract-v1",
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(
            semantic_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert provider.identity == {
        "command_processor_version": "command-processor-v3",
        "command_semantic_version": "processor-contract-v1",
        "command_semantic_sha256": f"sha256:{semantic_sha256}",
        "identity_file:0:path": str(first.resolve()),
        "identity_file:0:sha256": (
            "sha256:" + hashlib.sha256(first.read_bytes()).hexdigest()
        ),
        "identity_file:1:path": str(second.resolve()),
        "identity_file:1:sha256": (
            "sha256:" + hashlib.sha256(second.read_bytes()).hexdigest()
        ),
    }


def test_identity_tracks_semantic_command_config_but_not_retry_policy(
    tmp_path: Path,
) -> None:
    baseline = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="processor-v1",
            argv=("annotate", "--fps", "2"),
            timeout_s=10,
            retryable_exit_codes=(75,),
        ),
        tmp_path,
    ).identity
    changed_argv = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="processor-v1",
            argv=("annotate", "--fps", "10"),
            timeout_s=10,
            retryable_exit_codes=(75,),
        ),
        tmp_path,
    ).identity
    changed_retry_policy = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="processor-v1",
            argv=("annotate", "--fps", "2"),
            timeout_s=999,
            retryable_exit_codes=(2,),
        ),
        tmp_path,
    ).identity

    assert changed_argv["command_semantic_sha256"] != (
        baseline["command_semantic_sha256"]
    )
    assert changed_retry_policy == baseline


def test_artifact_scan_rejects_symlinks(tmp_path: Path) -> None:
    item = _item(tmp_path)
    context = _context(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    script = (
        "import json, pathlib, sys; "
        "out = pathlib.Path(sys.argv[1]); "
        "(out / 'output.jsonl').write_text(json.dumps({{'ok': True}}) + '\\n'); "
        "(out / 'unsafe-link').symlink_to(pathlib.Path(sys.argv[2]))"
    )
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(
                sys.executable,
                "-c",
                script,
                "{output_dir}",
                str(outside),
            )
        )
    )

    with pytest.raises(CommandProcessorError) as captured:
        _run_processor(provider, item, context)

    assert captured.value.kind == "unsafe_artifact"
    assert captured.value.retryable is False


def test_missing_media_template_is_a_non_retryable_error(
    tmp_path: Path,
) -> None:
    item = _item(tmp_path)
    context = _context(tmp_path)
    provider = CommandProcessorProvider(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(sys.executable, "-c", "pass", "{media_LEFT_WRIST_RGB}"),
        )
    )

    with pytest.raises(CommandProcessorError) as captured:
        _run_processor(provider, item, context)

    assert captured.value.kind == "template"
    assert captured.value.retryable is False


@pytest.mark.parametrize(
    "values",
    (
        {"argv": ()},
        {"argv": ("command", "{unknown}")},
        {"argv": ("command", "{item_id!r}")},
        {"argv": ("command",), "result_path": Path("../output.jsonl")},
        {"argv": ("command",), "result_path": Path("/tmp/output.jsonl")},
        {"argv": ("command",), "unexpected": True},
    ),
)
def test_config_rejects_unsafe_or_unknown_values(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CommandProcessorConfig.model_validate(
            {"semantic_version": "test-v1", **values}
        )


def test_config_requires_explicit_semantic_version() -> None:
    with pytest.raises(ValidationError, match="semantic_version"):
        CommandProcessorConfig.model_validate({"argv": ("command",)})


def test_exports_command_processor_plugin() -> None:
    assert PLUGIN.config_model is CommandProcessorConfig
    provider = PLUGIN.factory(
        CommandProcessorConfig(
            semantic_version="test-v1",
            argv=(sys.executable, "--version"),
        ),
        Path.cwd(),
    )
    assert isinstance(provider, CommandProcessorProvider)
