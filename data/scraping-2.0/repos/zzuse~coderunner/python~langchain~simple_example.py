
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os
import time

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
chat_model = ChatOpenAI(openai_api_key=api_key)
# result = chat_model.predict("hi!")
# print(result)

conversation = ConversationChain(llm=chat_model, verbose=True)
conversation.predict(input="Hi There!")
time.sleep(20)
conversation.predict(input="I'm doing well! Just having a conversation with AI.")
time.sleep(20)
conversation.predict(input="What was the first thing I said to you?")
time.sleep(20)
result = conversation.predict(input="what is an alternative phrase for the first thing I said to you?")
print(result)