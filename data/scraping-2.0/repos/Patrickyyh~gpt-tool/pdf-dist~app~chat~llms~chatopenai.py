from langchain.chat_models import ChatOpenAI


def build_llm(chat_args):
    return ChatOpenAI(model="gpt-4-1106-preview" , streaming= chat_args.streaming)


