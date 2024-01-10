# refer:https://python.langchain.com/docs/integrations/llms/google_ai    ...https://python.langchain.com/docs/integrations/chat/google_generative_ai
#original google AI: https://ai.google.dev/tutorials/python_quickstart?hl=en

import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI

from getpass import getpass


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
    #api_key = getpass() #"AIzaSyAG9M9CFujKMKgUD1z_n-hLGozCs90ZTJ8"
else:
    api_key = os.environ["GOOGLE_API_KEY"]
    print("GOOGLE_API_KEY already set to:", api_key) 
api_key = os.environ["GOOGLE_API_KEY"]
print("GOOGLE_API_KEY=>", api_key)
    
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)
print(
    llm.invoke(
        "What are some of the pros and cons of Python as a programming language?"
    )
)

llm = ChatGoogleGenerativeAI(model="gemini-pro")
result = llm.invoke("Write a ballad about LangChain")
print("ChatGoogleGenerativeAI.result=>", result.content)