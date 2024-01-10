from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

resp = chat.invoke(
    [
        HumanMessage(content="おいしいステーキの焼き方を教えて"),
    ],
)
print(resp.content)
