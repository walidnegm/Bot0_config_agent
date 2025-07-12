def call(args: dict) -> dict:
    msg = args.get("message", "")
    return {
        "status": "success",
        "message": msg
    }

