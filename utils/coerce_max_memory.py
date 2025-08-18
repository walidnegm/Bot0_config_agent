"""utils/coerce_max_memory.py"""

import re

_GIB_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*GiB\s*$", re.IGNORECASE)


def coerce_max_memory(mm: dict | None) -> dict | None:
    """
    Normalize YAML-provided max_memory for HF/Accelerate:
      - Keys: "0" -> 0, keep "cpu" as "cpu", strip spaces, lowercase.
      - Values: allow '3.5GiB' (canonical). Optionally accept 'GB' and convert to 'GiB'.
    """
    if not mm:
        return mm

    fixed = {}
    for k, v in mm.items():
        # --- normalize key ---
        key = str(k).strip()
        key = key.lower()
        if key.isdigit():
            key = int(key)
        elif key != "cpu":
            # allow 'cuda:0' -> 0
            if key.startswith("cuda:") and key[5:].isdigit():
                key = int(key[5:])
            else:
                raise ValueError(
                    f"max_memory key must be GPU index or 'cpu', got: {k!r}"
                )

        # --- normalize value ---
        val = str(v).strip()
        m = _GIB_RE.match(val)
        if m:
            # already GiB; normalize casing
            val = f"{m.group(1)}GiB"
        else:
            # optional: accept GB and convert approx to GiB (divide by 1.073741824)
            if val.lower().endswith("gb"):
                num = float(val[:-2].strip())
                gib = num / 1.073741824
                val = f"{gib:.3f}GiB"
            else:
                raise ValueError(
                    f"max_memory value must look like '3.5GiB' (or GB), got: {v!r}"
                )

        fixed[key] = val
    return fixed
