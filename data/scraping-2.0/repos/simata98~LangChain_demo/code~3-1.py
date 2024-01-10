from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI()

content = "코딩"

result = chat_model.predict(content + "에 대한 시를 써줘")
print(result)