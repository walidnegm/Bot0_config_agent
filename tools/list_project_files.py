import os

def call(args: dict) -> dict:
    root = args.get("root", ".")
    file_paths = []

    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            full_path = os.path.relpath(os.path.join(dirpath, f), start=root)
            file_paths.append(full_path)

    return {
        "status": "success",
        "files": file_paths,
        "message": f"Found {len(file_paths)} files under '{root}'"
    }

