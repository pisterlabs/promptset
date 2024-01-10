from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI()
chat_model = ChatOpenAI()

llm_result = llm.predict("say hi!")
chat_model_result = chat_model.predict("say hi!")

print(f"LLM: {repr(llm_result)}")
print(f"Chat Model: {repr(chat_model_result)}")
