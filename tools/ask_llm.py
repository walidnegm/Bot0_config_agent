from agent.llm_manager import ask_llm

def talk_llm(query, role="helper", temperature=0.2):
    try:
        response = ask_llm(query, temperature=temperature, role=role)
        return {
            "status": "ok",
            "message": response
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

