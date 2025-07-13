def call(**kwargs) -> dict:
    msg = kwargs.get("message", "")
    return {
        "status": "success",
        "message": msg
    }

