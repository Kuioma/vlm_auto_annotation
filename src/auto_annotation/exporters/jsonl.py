import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from pydantic_core import to_jsonable_python


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    normalized = [to_jsonable_python(record) for record in records]
    normalized.sort(key=lambda record: record["video_id"])
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in normalized:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
