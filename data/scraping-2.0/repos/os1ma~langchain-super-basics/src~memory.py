from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, SQLChatMessageHistory

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0)

# 会話履歴の準備
history = SQLChatMessageHistory(
    session_id="123",
    connection_string="sqlite:///memory.sqlite",
)
memory = ConversationBufferMemory(chat_memory=history)

# chainを作成
chain = ConversationChain(llm=llm, memory=memory)

# 繰り返し会話を行う
while True:
    user_message = input("You: ")
    ai_message = chain.run(user_message)
    print(f"AI:  {ai_message}")
