"""
Single-file saver with atomic writes, simple path policy, and flexible formats.

- Writes exactly one file per call.
- Supports json / ndjson / txt / yaml (yaml is optional; requires PyYAML).
- Optional gzip compression (adds .gz if not present).
- Enforces base_dir pinning and prevents path traversal.
- Returns path, bytes_written, sha256, and timestamp.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Optional, Union
from datetime import datetime, timezone
import io
import json
import gzip
import os
import hashlib
import tempfile

try:
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:
    yaml = None  # type: ignore
    _HAS_YAML = False

from bot0_config_agent.tools.configs.tool_models import (
    SaveToFileInput,
    SaveToFileResult,
)


def _is_relative_to(child: Path, parent: Path) -> bool:
    """Py<3.9 compat for Path.is_relative_to()."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _serialize_content(fmt: str, content: Any, *, encoding: str) -> bytes:
    fmt = fmt.lower()
    if fmt == "json":
        if isinstance(content, str):
            # If caller gives a string for JSON, try to parse to ensure validity
            try:
                obj = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"format=json but content is not valid JSON: {e}"
                ) from e
        else:
            obj = content
        return json.dumps(
            obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False
        ).encode(encoding)

    if fmt == "ndjson":
        # Accept: list/iterable of dict-like or strings (already JSON lines)
        buf = io.StringIO()
        if isinstance(content, str):
            # assume it's already NDJSON text
            for line in content.splitlines():
                if line.strip():
                    buf.write(line.rstrip("\n") + "\n")
        else:
            if isinstance(content, dict):
                content = [content]
            if not isinstance(content, Iterable):
                raise ValueError(
                    "format=ndjson requires an iterable (list/iterable) or a NDJSON string"
                )
            for item in content:
                if isinstance(item, str):
                    buf.write(item.rstrip("\n") + "\n")
                else:
                    buf.write(json.dumps(item, ensure_ascii=False) + "\n")
        return buf.getvalue().encode(encoding)

    if fmt == "txt":
        if isinstance(content, (bytes, bytearray)):
            return bytes(content)
        return str(content).encode(encoding)

    if fmt == "yaml":
        if not _HAS_YAML:
            raise RuntimeError("format=yaml requires PyYAML to be installed")
        if isinstance(content, str):
            # If given YAML text, pass it through
            return content.encode(encoding)
        return yaml.safe_dump(content, sort_keys=False).encode(encoding)  # type: ignore

    if fmt == "raw":
        if isinstance(content, (bytes, bytearray)):
            return bytes(content)
        raise ValueError("format=raw requires bytes/bytearray content")

    raise ValueError(f"Unsupported format: {fmt}")


def _apply_compression_if_needed(
    data: bytes,
    target_path: Path,
    compress: str,
) -> tuple[bytes, Path]:
    c = (compress or "none").lower()
    if c == "none":
        return data, target_path
    if c == "gzip":
        gz_bytes = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=gz_bytes) as gz:
            gz.write(data)
        # ensure .gz suffix exists
        if not str(target_path).endswith(".gz"):
            target_path = target_path.with_suffix(target_path.suffix + ".gz")
        return gz_bytes.getvalue(), target_path
    raise ValueError(f"Unsupported compress option: {compress}")


def _atomic_write_bytes(target_path: Path, data: bytes, *, perm: int = 0o640) -> int:
    _ensure_parent_dir(target_path)
    # temp file in the same directory to ensure atomic rename across filesystems
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=str(target_path.parent),
        prefix=".tmp-",
        suffix=".part",
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    os.replace(str(tmp_path), str(target_path))
    try:
        os.chmod(target_path, perm)
    except Exception:
        # permissions setting is best-effort (platform dependent)
        pass
    return len(data)


def save_to_file(params: Union[SaveToFileInput, dict]) -> SaveToFileResult:
    """
    Tool entrypoint. Validates inputs, enforces path policy, serializes, and writes atomically.
    """
    inp = params if isinstance(params, SaveToFileInput) else SaveToFileInput(**params)

    # Resolve base_dir and target path
    base_dir = Path(inp.base_dir or ".").resolve()
    # Caller supplies either a relative "path" (recommended) or absolute path (blocked by policy)
    target_rel = Path(inp.path)

    if target_rel.is_absolute():
        raise ValueError(
            "Absolute paths are not allowed. Provide a path relative to base_dir."
        )

    target_path = (base_dir / target_rel).resolve()
    if not _is_relative_to(target_path, base_dir):
        raise ValueError("Path traversal detected. The resolved path escapes base_dir.")

    # Serialize content
    data = _serialize_content(inp.format, inp.content, encoding=inp.encoding or "utf-8")

    # Compression (optional)
    data, target_path = _apply_compression_if_needed(
        data, target_path, inp.compress or "none"
    )

    # Mode handling
    mode = (inp.mode or "atomic").lower()
    if target_path.exists():
        if mode == "fail_if_exists":
            raise FileExistsError(
                f"Target exists and mode=fail_if_exists: {target_rel}"
            )
        # atomic -> overwrite safely; append supported only for line-oriented/text formats
        if mode == "append":
            # append is not allowed for json/yaml/raw (non-line-safe)
            if inp.format not in ("txt", "ndjson"):
                raise ValueError(
                    "mode=append is only supported for txt or ndjson formats"
                )
            _ensure_parent_dir(target_path)
            with open(target_path, "ab") as f:
                f.write(data)
            checksum = _sha256_bytes(data)  # checksum of appended chunk
            return SaveToFileResult(
                path=str(target_path),
                bytes_written=len(data),
                checksum=checksum,
                created_at=datetime.now(timezone.utc).isoformat(),
            )

    # Atomic write (default and for overwrite)
    written = _atomic_write_bytes(target_path, data)

    # Integrity
    checksum = _sha256_bytes(data)

    return SaveToFileResult(
        path=str(target_path),
        bytes_written=written,
        checksum=checksum,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
