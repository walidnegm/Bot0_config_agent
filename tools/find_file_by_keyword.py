import os

def find_file_by_keyword(**kwargs):
    keywords = kwargs.get("keywords")
    root = kwargs.get("root", os.getcwd())

    if not keywords:
        return {"status": "error", "message": "Missing required parameter: keywords"}

    if isinstance(keywords, str):
        keywords = [keywords]

    keywords = [k.lower() for k in keywords]
    matches = []

    for dirpath, _, files in os.walk(root):
        for f in files:
            if any(k in f.lower() for k in keywords):
                matches.append(os.path.relpath(os.path.join(dirpath, f), start=root))

    return {
        "status": "ok",
        "message": f"Found {len(matches)} file(s) matching keywords: {', '.join(keywords)}",
        "matches": matches
    }

