# Memoryの実験
# めんどうなコードを書かなくても会話のログを記録してくれる
from langchain import OpenAI, ConversationChain
import os

# AI-MOPが対応してない機能を使うと普通に動かない
os.environ["OPEN_API_KEY"] = os.environ.get('INIAD_OPENAI_API_KEY')
os.environ["OPEN_API_BASE"] = "https://api.openai.iniad.org/api/v1/"


llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

# 最初の会話
output = conversation.run("私の名前はかめさんです。よろしくね")
# 二番目の会話
output = conversation.run("私は誰ですか？")

# 二番目の会話に最初の会話の文脈が反映されているはず
print(output)
