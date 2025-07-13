from agent.llm_manager import ask_llm

def talk_llm(**kwargs):
    query = kwargs.get("query")
    role = kwargs.get("role", "helper")
    temperature = kwargs.get("temperature", 0.2)

    if not query:
        return {
            "status": "error",
            "message": "Missing required parameter: query"
        }

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

