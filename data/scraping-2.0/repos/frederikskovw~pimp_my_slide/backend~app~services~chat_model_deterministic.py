from langchain.chat_models import ChatOpenAI

def chat_model_deterministic(key, model_name):
    return ChatOpenAI(
        openai_api_key=key,
        temperature=0,
        model_name=model_name,
    )