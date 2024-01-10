from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
result = llm.predict("自己紹介してください！")
print(result)
