"""
bot0_config_agent/tools/tool_scripts/select_files.py

Utility to filter, sort, and slice a list of file paths without reading file contents.

Features
--------
- De-duplicates input while preserving order
- Filters by file extension (include/exclude)
- Filters by filename substring (include/exclude)
- Sorts by:
    - "name"  : case-insensitive filename
    - "path"  : case-insensitive full path
    - "mtime" : last modified time (filesystem I/O)
    - "ctime" : creation time on Windows, metadata change time on Unix (filesystem I/O)
    - "atime" : last access time (filesystem I/O)
- Offset/limit pagination or "tail N"
- Hard cap to prevent oversized outputs
- Robust error handling (missing files during date sorts are pushed to the end)

Return shape
------------
{
  "status": "success" | "error",
  "message": "...",
  "result": {
    "files": [<str>, ...],
    "total": <int>,         # total after filtering, before slicing/cap
    "selected": <int>,      # number returned
    "applied": {
      "tail": <bool>,
      "offset": <int>,
      "limit": <int|None>,
      "sort_by": <str|None>,
      "order": "asc"|"desc",
      "include_ext": [<str>, ...],
      "exclude_ext": [<str>, ...],
      "include_name": [<str>, ...],
      "exclude_name": [<str>, ...],
      "hard_cap": <int>
    }
  }
}
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Iterable
from pathlib import Path


logger = logging.getLogger(__name__)


def _norm_ext(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _match_name(name: str, needles: List[str]) -> bool:
    lname = name.lower()
    return any(n.lower() in lname for n in needles)


def _coerce_files_to_paths(v: Any) -> List[Path]:
    def to_path(x: Any) -> Optional[Path]:
        if isinstance(x, Path):
            return x
        if isinstance(x, str) and x.strip():
            return Path(x.strip()).expanduser()
        return None

    if v is None:
        return []
    if isinstance(v, (str, Path)):
        p = to_path(v)
        return [p] if p else []
    if isinstance(v, Iterable):
        out: List[Path] = []
        for item in v:
            p = to_path(item)
            if p:
                out.append(p)
        return out
    return []


def _safe_stat(p: Path):
    try:
        return p.stat()
    except Exception:
        return None


def select_files(**kwargs: Any) -> Dict[str, Any]:
    """
    Parameters (kwargs)
    -------------------
    files : list[str|Path]
    include_ext : list[str]        # e.g., [".py", "md"]
    exclude_ext : list[str]
    include_name : list[str]
    exclude_name : list[str]
    sort_by : str | None           # "filename"|"name"|"path"|"mtime"|"ctime"|"atime"
    order : "asc" | "desc"         # default "asc"
    offset : int                   # ignored when tail=True
    limit : int | None
    tail : bool                    # if True, take last N items where N=limit
    hard_cap : int                 # default 100; ≤0 disables

    Returns
    -------
    dict with keys: status, message, result
    result: { files: List[str], total: int, selected: int, applied: {...} }
    """
    try:
        # 1) Normalize inputs → List[Path]
        files_in = kwargs.get("files") or []
        if not isinstance(files_in, list):
            return {
                "status": "error",
                "message": "'files' must be a list",
                "result": None,
            }
        paths: List[Path] = _coerce_files_to_paths(files_in)

        # De-dup (preserve order) using string repr for stable hashing
        seen: set[str] = set()
        uniq: List[Path] = []
        for p in paths:
            s = str(p)
            if s not in seen:
                seen.add(s)
                uniq.append(p)
        paths = uniq

        # 2) Filters
        include_ext = [
            _norm_ext(e)
            for e in (kwargs.get("include_ext") or [])
            if isinstance(e, str)
        ]
        exclude_ext = [
            _norm_ext(e)
            for e in (kwargs.get("exclude_ext") or [])
            if isinstance(e, str)
        ]
        include_name = [str(x) for x in (kwargs.get("include_name") or [])]
        exclude_name = [str(x) for x in (kwargs.get("exclude_name") or [])]

        if include_ext:
            paths = [p for p in paths if p.suffix.lower() in include_ext]
        if exclude_ext:
            paths = [p for p in paths if p.suffix.lower() not in exclude_ext]

        if include_name:
            paths = [p for p in paths if _match_name(p.name, include_name)]
        if exclude_name:
            paths = [p for p in paths if not _match_name(p.name, exclude_name)]

        # 3) Sorting
        sort_by = (kwargs.get("sort_by") or "").strip().lower() or None
        # accept both "filename" and legacy "name"
        if sort_by == "name":
            sort_by = "filename"

        order = (kwargs.get("order") or "asc").strip().lower()
        order = "desc" if order == "desc" else "asc"
        reverse = order == "desc"

        if sort_by == "filename":
            paths.sort(key=lambda p: p.name.lower(), reverse=reverse)
        elif sort_by == "path":
            paths.sort(key=lambda p: str(p).lower(), reverse=reverse)
        elif sort_by in {"mtime", "ctime", "atime"}:
            desc = order == "desc"

            def _time_key(p: Path):
                st = _safe_stat(p)
                if st is None:
                    return (1, 0.0)  # push missing/unreadable to end
                t = (
                    st.st_mtime
                    if sort_by == "mtime"
                    else st.st_ctime if sort_by == "ctime" else st.st_atime
                )
                return (0, -t if desc else t)

            # Use a composite key so missing files stay last in both orders
            paths.sort(key=_time_key)
        # else: no sort

        total = len(paths)

        # 4) Slicing
        tail = bool(kwargs.get("tail", False))
        limit = kwargs.get("limit")
        limit = int(limit) if isinstance(limit, int) else None
        offset = int(kwargs.get("offset", 0)) if not tail else 0

        if tail:
            sel = paths[-limit:] if (isinstance(limit, int) and limit > 0) else paths[:]
            offset_used = 0
        else:
            start = max(0, offset)
            end = start + limit if (isinstance(limit, int) and limit > 0) else None
            sel = paths[start:end]
            offset_used = start

        # 5) Hard cap
        hard_cap = int(kwargs.get("hard_cap", 100))
        if hard_cap > 0 and len(sel) > hard_cap:
            sel = sel[:hard_cap]

        # 6) Message + result
        msg = f"Selected {len(sel)} of {total}"
        if sort_by:
            msg += f" (sorted by {sort_by} {order})"
        if tail:
            msg += " (tail)"
        else:
            msg += (
                f" (offset {offset_used}"
                + (f", limit {limit}" if isinstance(limit, int) else "")
                + ")"
            )

        return {
            "status": "success",
            "message": msg,
            "result": {
                "files": [str(p) for p in sel],
                "total": total,
                "selected": len(sel),
                "applied": {
                    "tail": tail,
                    "offset": offset_used,
                    "limit": limit,
                    "sort_by": sort_by,
                    "order": order,
                    "include_ext": include_ext,
                    "exclude_ext": exclude_ext,
                    "include_name": include_name,
                    "exclude_name": exclude_name,
                    "hard_cap": hard_cap,
                },
            },
        }

    except Exception as e:
        return {"status": "error", "message": str(e), "result": None}
