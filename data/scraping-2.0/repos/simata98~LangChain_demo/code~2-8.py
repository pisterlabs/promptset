from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI()
result = chat_model.predict("hi")
print(result)