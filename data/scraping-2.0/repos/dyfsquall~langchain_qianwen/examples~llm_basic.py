from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_qianwen import Qwen_v1


if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-turbo",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    question = "你好, 讲个50字的笑话吧"
    resp = llm(question)
    print("type: ", type(resp))
    print("resp: ", resp)
