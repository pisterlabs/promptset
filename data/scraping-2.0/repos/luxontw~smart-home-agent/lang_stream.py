from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.0,
    request_timeout=60,
)
for chunk in chat.stream("Write me a song about goldfish on the moon"):
    print(chunk.content, end="", flush=True)
