from langchain.chat_models import ChatOpenAI

from app.core.config import settings

chat_process_conversation = ChatOpenAI(
    temperature=0,
    openai_api_key=settings.OPEN_AI_API_KEY,
    model=settings.LLM_MODEL_CONV,
)
chat_create_response_template = ChatOpenAI(
    temperature=0,
    openai_api_key=settings.OPEN_AI_API_KEY,
    model=settings.LLM_MODEL_TEMP,
)
