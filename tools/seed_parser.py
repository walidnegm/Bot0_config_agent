def seed_parser(**kwargs):
    file = kwargs.get("file")
    if not file:
        return { "error": "Missing required parameter: file" }

    try:
        with open(file, "r") as f:
            lines = f.readlines()
        kv_pairs = dict(line.strip().split("=", 1) for line in lines if "=" in line)
        return { "parsed": kv_pairs }
    except Exception as e:
        return { "status": "error", "message": str(e) }

