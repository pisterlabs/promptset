from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

from robojudge.utils.settings import settings

standard_llm = (
    AzureChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        openai_api_version=settings.OPENAI_API_VERSION,
        openai_api_type=settings.OPENAI_API_TYPE,
        deployment_name=settings.GPT_MODEL_NAME,
        temperature=0,
    )
    if settings.OPENAI_API_TYPE == "azure"
    else ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model=settings.GPT_MODEL_NAME,
        temperature=0,
    )
)

advanced_llm = (
    AzureChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        openai_api_version=settings.OPENAI_API_VERSION,
        openai_api_type=settings.OPENAI_API_TYPE,
        deployment_name=settings.AUTO_EVALUATOR_NAME,
        temperature=0,
    )
    if settings.OPENAI_API_TYPE == "azure"
    else ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model=settings.AUTO_EVALUATOR_NAME,
        temperature=0,
    )
)
