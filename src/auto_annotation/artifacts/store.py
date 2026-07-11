import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python


class ArtifactReference(BaseModel):
    path: Path
    sha256: str


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _validate_destination_parent(self, parent: Path) -> None:
        resolved_root = self.root.resolve(strict=False)
        resolved_parent = parent.resolve(strict=False)
        if not resolved_parent.is_relative_to(resolved_root):
            raise ValueError(
                f"artifact destination escapes artifact root: {parent}"
            )
        current = self.root
        for component in parent.relative_to(self.root).parts:
            current /= component
            if current.is_symlink():
                raise ValueError(
                    f"artifact destination has symlink ancestor: {current}"
                )

    def write_json(
        self,
        run_id: str,
        video_id: str,
        stage: str,
        payload: Any,
    ) -> ArtifactReference:
        for component in (run_id, video_id, stage):
            if component in {"", ".", ".."} or "/" in component or "\\" in component:
                raise ValueError(f"unsafe artifact component: {component}")

        destination = self.root / run_id / video_id / f"{stage}.json"
        self._validate_destination_parent(destination.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._validate_destination_parent(destination.parent)
        data = json.dumps(
            to_jsonable_python(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        return ArtifactReference(
            path=destination,
            sha256=f"sha256:{hashlib.sha256(data).hexdigest()}",
        )

    def read_json(self, reference: ArtifactReference) -> Any:
        return json.loads(reference.path.read_text(encoding="utf-8"))
