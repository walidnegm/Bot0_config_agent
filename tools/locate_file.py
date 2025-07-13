# tools/locate_file.py

import os

def locate_file(filename: str):
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

