from langchain.memory import ConversationBufferWindowMemory
import os
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

memory = ConversationBufferWindowMemory( k=1)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})

api_key = os.environ.get('OPEN_AI_KEY')

chain = ConversationChain(
    llm=OpenAI(temperature=0), 
    memory=memory, 
    verbose=True
)

history = []

while True:
    query = input("Enter Your Query:")
    print(chain.predict(input=query))