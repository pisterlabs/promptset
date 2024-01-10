from dotenv import load_dotenv, find_dotenv

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

"""
Load OpenAI API key 
"""
_ = load_dotenv(find_dotenv())


"""
Language Model
"""
llm = OpenAI(temperature=0.0)
prompt = "Apa persyaratan untuk menjadi seorang profesor di Indonesia?"
response = llm(prompt)
print(f"[Language] Prompt: {prompt}")
print(f"[Language] Response: {response}")


"""
Chat Model
"""
chat_prompt = [
    SystemMessage(content="Kamu adalah seorang administrator pendidikan tinggi di Indonesia yang sangat mengerti mengenai karir akademik di Indonesia"),
    HumanMessage(content=prompt),
]
chat_llm = ChatOpenAI(temperature=0.0)
chat_response = chat_llm(chat_prompt)
print(f"[Chat] Prompt: {prompt}")
print(f"[Chat] Response: {chat_response.content}")




