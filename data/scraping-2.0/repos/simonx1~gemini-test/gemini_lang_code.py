from langchain.chat_models import ChatVertexAI
from langchain.prompts import ChatPromptTemplate

chat = ChatVertexAI(
    model_name="codechat-bison", max_output_tokens=1000, temperature=0.5
)

message = chat.invoke("Write a Python function to identify all prime numbers")
print(message.content)

