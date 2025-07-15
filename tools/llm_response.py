# tools/llm_response.py
import os

def llm_response(prompt: str) -> dict:
    from agent import llm_openai
    from agent.llm_manager import LLMManager

    try:
        use_openai = bool(int(os.getenv("USE_OPENAI", "0")))
        response = llm_openai.generate(prompt) if use_openai else LLMManager().generate(prompt)
        return {
            "status": "ok",
            "message": response,
            "result": {"text": response}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

