from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory


class Conversation:
    def __init__(self, num_of_round=10):
        self.conversation = ConversationChain(
            llm=ChatOpenAI(temperature=0.5, max_tokens=2048),
            memory=ConversationBufferWindowMemory(k=num_of_round),
            verbose=True,
        )

    def ask(self, question: str) -> str:
        return self.conversation.predict(input=question)


if __name__ == '__main__':
    myPrompt = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:1. 你的回答必须是中文2. 回答限制在100个字以内"""
    conv = Conversation(10)
    answer = conv.ask('你好')
    print(answer)
    answer = conv.ask('给我讲个笑话')
    print(answer)
