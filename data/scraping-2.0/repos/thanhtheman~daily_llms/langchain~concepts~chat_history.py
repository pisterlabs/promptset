from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)

history = ChatMessageHistory()
history.add_ai_message("hello")
history.add_user_message("What is the capital of Canada")
ai_response = llm(history.messages)
history.add_ai_message(ai_response)
print(history.messages)