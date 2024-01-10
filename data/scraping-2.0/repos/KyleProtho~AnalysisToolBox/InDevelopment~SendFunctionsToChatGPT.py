# Load packages
from dotenv import load_dotenv
import json
import openai
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import DocArrayInMemorySearch

# Declare functions
def SendFunctionsToChatGPT(list_of_functions,
                           user_prompt,
                           system_message="You are a helpful assistant.",
                           # LLM parameters
                           openai_api_key=None,
                           temperature=0.0,
                           chat_model_name="gpt-3.5-turbo-0613",
                           function_calling_mode="auto",
                           verbose=True):
    # Ensure that function_calling_mode is auto or none, or a dictionary
    if function_calling_mode not in ["auto", "none"] or type(function_calling_mode) != dict:
        raise ValueError("function_calling_mode must be either 'auto' or 'none', or a dictionary of function names and their arguments.")
    
    # If OpenAI API key is not provided, then try to load from .env file
    if openai_api_key is None:
        load_dotenv()
        try:
            openai_api_key = os.environ['OPENAI_API_KEY']
        except:
            raise ValueError("No API key provided and no .env file found. If you need a OpenAI API key, visit https://platform.openai.com/")
    
    # Create SystemMessage and HumanMessage objects
    system_msg = SystemMessage(content=system_message)
    user_msg = HumanMessage(content=user_prompt)
    
    # Create messages list to send to chat model
    messages = [system_msg, user_msg]
    
    # Send functions and prompt to chat model
    response = openai.ChatCompletion.create(
        model=chat_model_name,
        messages=messages,
        functions=list_of_functions,
        function_call=function_calling_mode
    )
    
    # Print the response
    
    
