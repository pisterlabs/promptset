from langchain.chat_models import ChatOpenAI


def build_llm(chat_args, model_name):
    return ChatOpenAI(streaming=chat_args.streaming, verbose=True,
                      model_name=model_name)


def build_condense_question_chain_llm(chat_args):
    return ChatOpenAI(streaming=False, verbose=True)
