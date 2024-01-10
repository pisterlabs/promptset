from langchain.chat_models import ChatVertexAI
from langchain.prompts import ChatPromptTemplate

chat = ChatVertexAI(
    model_name="gemini-pro", max_output_tokens=8192, temperature=0.2
)

message = chat.invoke("What is your name?")
print(message.content)

