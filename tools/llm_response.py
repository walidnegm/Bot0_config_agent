def llm_response(**kwargs):
    prompt = kwargs.get("prompt", "")
    if not prompt:
        return {"status": "error", "message": "Missing prompt"}

    from agent.llm_manager import LLMManager
    llm = LLMManager()

    try:
        response = llm.generate(prompt=prompt.strip(), temperature=0.1)
        return {
            "status": "ok",
            "message": response,
            "result": response
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"LLM error: {e}"
        }

