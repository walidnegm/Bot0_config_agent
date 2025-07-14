def make_virtualenv(**kwargs):
    import venv
    import os

    path = kwargs.get("path")
    if not path:
        return {"status": "error", "message": "Missing required parameter: path"}

    try:
        os.makedirs(path, exist_ok=True)
        venv.create(path, with_pip=True)
        return {"status": "ok", "message": f"Virtualenv created at {path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

