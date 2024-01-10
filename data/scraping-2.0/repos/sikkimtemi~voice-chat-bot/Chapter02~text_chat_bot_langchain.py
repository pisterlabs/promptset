from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# プロンプトテンプレートの準備
template = """あなたは猫のキャラクターとして振る舞うチャットボットです。
制約:
- 簡潔な短い文章で話します
- 語尾は「…にゃ」、「…にゃあ」などです
- 質問に対する答えを知らない場合は「知らないにゃあ」と答えます
- 名前はクロです
- 好物はかつおぶしです"""

# プロンプトの準備
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# LangChainのLarge Language Model (LLM)を設定
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# メモリの設定
memory = ConversationBufferMemory(return_messages=True)

# チャットボットの作成
conversation = ConversationChain(llm=llm, verbose=True, prompt=prompt, memory=memory)

# ユーザーからのメッセージを受け取り、それに対する応答を生成
while True:
    user_message = input("あなたのメッセージを入力してください: \n")
    response = conversation.predict(input=user_message)
    print("チャットボットの回答: \n" + response)
