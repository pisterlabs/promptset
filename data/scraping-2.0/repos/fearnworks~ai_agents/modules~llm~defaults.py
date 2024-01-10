import os 
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')



def get_default_cloud_chat_llm():
    """
    Returns a default LLM instance with the OpenAI API key set in the environment.

    Returns:
        OpenAI: A new OpenAI instance.
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)
    return llm

def get_default_cloud_completion_llm():
    """
    Returns a default LLM instance with the OpenAI API key set in the environment.

    Returns:
        OpenAI: A new OpenAI instance.
    """
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    return llm

def get_default_local_llm():
    """
    Coming soon! 
    """
    pass