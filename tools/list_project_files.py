import os

def list_project_files(**kwargs) -> dict:
    root = kwargs.get("root") or os.getcwd()  # ✅ Absolute fallback
    print(f"[list_project_files] 🔍 Scanning directory: {root}")

    if not os.path.exists(root):
        return {
            "status": "error",
            "message": f"Directory '{root}' does not exist."
        }

    file_paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            full_path = os.path.relpath(os.path.join(dirpath, f), start=root)
            file_paths.append(full_path)

    return {
        "status": "success",
        "files": file_paths,
        "message": f"Found {len(file_paths)} file(s) under '{root}'"
    }

