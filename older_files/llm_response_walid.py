import os
from agent.llm_manager import LLMManager
from agent.llm_openai import generate as openai_generate


def llm_response(**kwargs):
    prompt = kwargs.get("prompt", "")
    if not prompt:
        return {"status": "error", "message": "Missing prompt"}

    use_openai = os.environ.get("USE_OPENAI", "").lower() == "true"

    try:
        if use_openai:
            response = openai_generate(prompt.strip(), temperature=0.1)
        else:
            llm = LLMManager()
            response = llm.generate(prompt.strip(), temperature=0.1)

        return {"status": "ok", "message": response, "result": response}

    except Exception as e:
        return {"status": "error", "message": f"LLM error: {e}"}
