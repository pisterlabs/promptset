from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os
load_dotenv()

# different possible additions to the model
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     AIMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
def get_gpt_llm():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    chat_params = {
        "model": "gpt-3.5-turbo-16k", # Bigger context window
        "openai_api_key": OPENAI_API_KEY, # load from enviornment
        "temperature": 0.5, # To avoid pure copy-pasting from docs lookup
        "max_tokens": 8192,
        "streaming": True,
        "callbacks": [StreamingStdOutCallbackHandler()]
    }
    llm = ChatOpenAI(**chat_params)
    return llm

# api key is required
# def get_hugging_face_llm(repository_id: string):
#     model_parameters = {
#         "temperature": 0.5,
#         "max_length": 512
#     }
#     llm = HuggingFaceHub(repository_id, model_kwargs = model_parameters)