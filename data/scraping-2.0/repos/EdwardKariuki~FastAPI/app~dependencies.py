import openai
from openai import AzureOpenAI
from app.core.settings import settings
from tenacity import retry, stop_after_delay


def get_openai():
    # sets open API API KEY
    openai.api_key = settings.openai_api_key
    print("API_KEY:", settings.openai_api_key)
    return openai
openai_instance = get_openai()
exit()