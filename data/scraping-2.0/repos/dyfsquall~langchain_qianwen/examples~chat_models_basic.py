from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_qianwen import ChatQwen_v1
from langchain.schema import (
    HumanMessage,
)

if __name__ == "__main__":
    chat = ChatQwen_v1(
        model_name="qwen-turbo",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    chat([HumanMessage(content="介绍一下 Golang")])
