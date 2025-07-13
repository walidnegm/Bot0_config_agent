import os

def locate_file(**kwargs):
    matches = []
    search_roots = [os.path.expanduser("~"), os.getcwd()]

    for root in search_roots:
        for dirpath, _, files in os.walk(root):
            if filename in files:
                matches.append(os.path.join(dirpath, filename))

    return {
        "status": "ok",
        "message": f"Found {len(matches)} match(es).",
        "matches": matches
    }

