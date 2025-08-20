"""tools/transformation.py"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import json


# ---------------------------
# Small helpers (robust & duck-typed)
# ---------------------------


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # ToolResult-like? try attributes
    out = {}
    for k in (
        "status",
        "message",
        "result",
        "files",
        "summary",
        "path",
        "paths",
        "content",
        "contents",
    ):
        if hasattr(obj, k):
            out[k] = getattr(obj, k)
    # Fallback: allow Pydantic BaseModel .model_dump() if present
    if hasattr(obj, "model_dump"):
        try:
            out.update(obj.model_dump())
        except Exception:
            pass
    return out or {"result": obj}


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _is_pathlike_str(s: Any) -> bool:
    return isinstance(s, str) and len(s.strip()) > 0


def _extract_paths_from_files_field(container: Any) -> List[str]:
    """
    Accepts many shapes:
      - ["a.py", "b.py"]
      - [{"file": "a.py", "content": "..."}, {"file": "b.py", "content": "..."}]
      - {"files": [...]} wrapper
    """
    if container is None:
        return []
    if isinstance(container, dict) and "files" in container:
        container = container["files"]

    items = _ensure_list(container)
    paths: List[str] = []

    for it in items:
        if _is_pathlike_str(it):
            paths.append(it)
        elif isinstance(it, dict):
            if "file" in it and _is_pathlike_str(it["file"]):
                paths.append(it["file"])
            elif "path" in it and _is_pathlike_str(it["path"]):
                paths.append(it["path"])
    return paths


def _extract_content_entries(container: Any) -> List[Dict[str, Any]]:
    """
    Pull out entries that look like {"file": <str>, "content": <str>} from
    common shapes:
      - {"result": {"files": [{"file":..., "content":...}, ...]}}
      - {"files": [{"file":..., "content":...}]}
      - {"summary": [{"file":..., "content":...}]}
      - [{"file":..., "content":...}]
    """
    if container is None:
        return []

    # Unwrap common wrapper keys
    if isinstance(container, dict):
        for key in ("result", "files", "summary", "contents", "content"):
            if key in container:
                inner = container[key]
                # result could be nested one more level (e.g., {"result": {"summary": [...]}})
                if isinstance(inner, dict) and (
                    "files" in inner or "summary" in inner or "contents" in inner
                ):
                    # merge inside-first key found
                    for k2 in ("files", "summary", "contents"):
                        if k2 in inner:
                            container = inner[k2]
                            break
                else:
                    container = inner
                break

    entries = _ensure_list(container)
    out: List[Dict[str, Any]] = []
    for it in entries:
        if isinstance(it, dict) and "content" in it:
            # accept even if "file" missing
            out.append(
                {
                    "file": it.get("file") or it.get("path") or "",
                    "content": it["content"],
                }
            )
    return out


def _pretty_join_text(chunks: Iterable[str], sep: str = "\n\n") -> str:
    return sep.join([c for c in chunks if isinstance(c, str) and c.strip()])


# ---------------------------
# Transform functions referenced in tool_transformation.json
# ---------------------------
def any_to_message(source_result: Any, **_) -> Dict[str, Any]:
    """
    source: * → target: echo_message
    Convert any payload into a string message for echo.
    Output: {"message": <str>}
    """
    if isinstance(source_result, str):
        msg = source_result
    else:
        try:
            msg = json.dumps(_as_dict(source_result), indent=2, ensure_ascii=False)
        except Exception:
            msg = str(source_result)
    return {"message": msg}


def contents_to_prompt(
    source_result: Any, instruction: Optional[str] = None, **_
) -> Dict[str, Any]:
    """
    source: read_files → target: llm_response_async
    Build a prompt from file contents. Flexible to accept multiple shapes.
    Output: {"prompt": <str>}
    """
    entries = _extract_content_entries(source_result)
    if not entries:
        # Try a permissive fallback: stringify the entire result
        body = json.dumps(_as_dict(source_result), indent=2, ensure_ascii=False)
        prompt = _pretty_join_text([instruction or "", "### DATA\n" + body]).strip()
        return {"prompt": prompt}

    parts: List[str] = []
    for e in entries:
        f = e.get("file") or ""
        c = e.get("content") or ""
        if f:
            parts.append(f"##### FILE: {f}\n{c}")
        else:
            parts.append(c)
    header = instruction or "Summarize or analyze the following file contents:"
    prompt = _pretty_join_text([header, *parts])
    return {"prompt": prompt}


def dir_to_files(source_result: Any, **_) -> Dict[str, Any]:
    """
    source: find_dir_structure → target: read_files | summarize_files
    Accepts a directory tree or any object that includes a flat 'files' listing.
    Strategy:
      1) If 'files' is present, use it.
      2) Else, walk a 'structure' tree and collect leaf file paths if provided.
      3) As a last resort, try to extract any 'path'/'file' fields present in lists.

    Output params for target tools that expect a file list:
      {"files": [<paths>]}
    """
    d = _as_dict(source_result)

    # Case 1: direct files
    files = _extract_paths_from_files_field(d.get("files"))
    if files:
        return {"files": files}

    # Case 2: maybe nested result.files
    files = _extract_paths_from_files_field(d.get("result"))
    if files:
        return {"files": files}

    # Case 3: try a structure tree at common keys
    tree = d.get("structure") or d.get("tree") or d.get("result")
    collected: List[str] = []

    def _walk(node: Any, prefix: str = ""):
        if not isinstance(node, dict):
            return
        ntype = node.get("type")
        name = node.get("name")
        children = node.get("children", [])
        if name and isinstance(name, str):
            path = f"{prefix}/{name}" if prefix else name
        else:
            path = prefix

        if ntype == "file" and path:
            collected.append(path)
        elif ntype == "directory":
            for ch in children or []:
                _walk(ch, path)

    _walk(tree)
    if collected:
        return {"files": collected}

    # Case 4: scan for any 'file'/'path' fields in common places
    possible = d.get("summary") or d.get("contents") or d.get("result")
    files = _extract_paths_from_files_field(possible)
    return {"files": files} if files else {"files": []}


def locate_to_select_files(source_result: Any, **_) -> Dict[str, Any]:
    """
    Prefer result.all_matches; fall back to result.path (single).
    Output: {"files": [...]}
    """
    d = _as_dict(source_result)
    r = d.get("result", d)
    files: list[str] = []

    if isinstance(r, dict):
        # prefer new "files", fallback to legacy "all_matches", then "path"
        for key in ("files", "all_matches"):
            am = r.get(key)
            if isinstance(am, list) and am:
                files = [s for s in am if _is_pathlike_str(s)]
                break
        if not files:
            p = r.get("first") or r.get("path")
            if _is_pathlike_str(p):
                files = [p]
    elif isinstance(source_result, str) and _is_pathlike_str(source_result):
        files = [source_result]

    return {"files": files}


def path_to_list(source_result: Any, **_) -> Dict[str, Any]:
    """
    source: locate_file → target: read_files
    Convert a single path into the list param expected by read_files.
    Output: {"files": [<path>]}
    """
    d = _as_dict(source_result)
    # Common shapes:
    # {"result": {"path": "a.py"}} or {"path": "a.py"} or {"file": "a.py"} or plain string
    if isinstance(source_result, str):
        path = source_result
    else:
        r = d.get("result", d)
        path = r.get("path") or r.get("file") or r.get("result")
        if isinstance(path, dict) and "path" in path:
            path = path["path"]
    return {"files": [path]} if _is_pathlike_str(path) else {"files": []}


def path_to_files(source_result: Any, **_) -> Dict[str, Any]:
    """
    source: locate_file → target: summarize_files
    Same behavior as path_to_list but keeps the param name aligned for summarize_files.
    Output: {"files": [<path>]}
    """
    return path_to_list(source_result)


def prompt_to_prompt(source_result: Any, **_) -> Dict[str, Any]:
    """
    source: preset_code_summary → target: llm_response_async
    Accepts either a plain string or dicts that contain 'prompt' at top-level or
        under 'result'.
    Output: {"prompt": <str>}
    """
    d = _as_dict(source_result)
    if isinstance(source_result, str):
        return {"prompt": source_result}

    # result may already be {"prompt": "..."} or contain it nested
    if isinstance(d.get("result"), dict) and "prompt" in d["result"]:
        return {"prompt": d["result"]["prompt"]}
    if "prompt" in d:
        return {"prompt": d["prompt"]}

    # last-ditch: stringify everything
    try:
        body = json.dumps(d, ensure_ascii=False, indent=2)
    except Exception:
        body = str(source_result)
    return {"prompt": body}


def summary_to_prompt(
    source_result: Any, instruction: Optional[str] = None, **_
) -> Dict[str, Any]:
    """
    source: summarize_files | summarize_config → target: llm_response_async
    Build a prompt from summaries.
    Output: {"prompt": <str>}
    """
    d = _as_dict(source_result)
    # Prefer 'summary' key; otherwise fall back to entries with 'content'
    summaries: List[Dict[str, Any]] = _ensure_list(d.get("summary"))
    if not summaries:
        summaries = _extract_content_entries(d)

    chunks: List[str] = []
    for s in summaries:
        f = ""
        c = s
        if isinstance(s, dict):
            f = s.get("file") or s.get("path") or ""
            c = s.get("content", s)
        text = f"##### SUMMARY for {f}\n{c}" if f else f"{c}"
        if isinstance(text, (str, bytes)):
            chunks.append(
                text if isinstance(text, str) else text.decode("utf-8", "ignore")
            )

    header = instruction or "Use the following summaries:"
    prompt = _pretty_join_text([header, *chunks])
    return {"prompt": prompt}


def path_to_file_param(source_result: Any, **_) -> Dict[str, Any]:
    """
    source: locate_file → target: seed_parser
    seed_parser expects {"file": "<path>"} (single file param).
    """
    d = _as_dict(source_result)
    if isinstance(source_result, str):
        path = source_result
    else:
        r = d.get("result", d)
        path = r.get("path") or r.get("file") or r.get("result")
        if isinstance(path, dict) and "path" in path:
            path = path["path"]
    return {"file": path} if _is_pathlike_str(path) else {"file": ""}
