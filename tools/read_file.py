# tools/read_file.py

def read_file(**kwargs):
    path = kwargs.get("path")
    if not path:
        return {
            "status": "error",
            "message": "Missing required parameter: path"
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "status": "ok",
            "message": f"Read contents of {path}",
            "result": content
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read file '{path}': {e}"
        }

