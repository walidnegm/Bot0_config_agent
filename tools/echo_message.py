import json

def echo_message(**kwargs):
    message = kwargs.get("message", "")
    result = {
        "status": "ok",
        "message": message
    }

    # Try to parse message if it's a serialized JSON string
    try:
        parsed = json.loads(message)
        if isinstance(parsed, dict) and "matches" in parsed:
            result["matches"] = parsed["matches"]
    except Exception:
        pass

    return result

