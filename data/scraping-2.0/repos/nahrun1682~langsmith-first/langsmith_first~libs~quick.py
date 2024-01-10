import os
import sys
 
#Pythonのパッケージを検索する対象パスのリストに、実行プログラムを起点にしたライブラリまでの相対パスを登録する。
#下記の例は、実行プログラム配下のフォルダを検索対象として追加している。
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#import settings
from langchain.chat_models import ChatOpenAI

def run():
    llm = ChatOpenAI()
    output = llm.predict("今日は暑すぎて、外に出たくない。")
    print(output)