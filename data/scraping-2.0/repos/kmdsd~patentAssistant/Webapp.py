#ライブラリの読み込み
import os
import platform
import streamlit as st
import shutil
import glob
import json
import chromadb
import langchain

from langchain.document_loaders  import TextLoader

# ﾗｲﾌﾞﾗﾘｰｲﾝﾎﾟｰﾄ
import sys
import time
import datetime

# Request
import requests
import urllib3

from src import vector_db
from src import join_llm
from src import patentEngineering

# ====================================================================
# 定数定義
# ====================================================================
# APIが社内(IIS)の場合
proxies = {
    "http": None,
    "https": None,
}

# APIｴﾝﾄﾞﾎﾟｲﾝﾄ(自社衝突回避ｻｼﾞｪｽﾄﾂｰﾙ)
avoid_url = "http://10.129.57.90:8100/vectorsearch"

# 変数定義
chunkSize = 1000
start = 0
end = 0
units = 1
wait = 20
title = '# 発明説明書作成支援ツール'
memo = '### -発明説明書の作成支援します- '
button2 = False
mk_db_flg = False

# DBが格納されているディレクトリの設定
dir_dict = {
    'input_dir': f'./input',
    'temp_dir': f'./temp',
    'persist_dir': f'./db'
}

# プロンプトの項目
prompt = {
    'name':'',
    'abstract':'',
    'issue':'',
    'method':''
}
# ====================================================================
# 関数定義
# ====================================================================
def get_similerIdea(query):
    # URLﾊﾟﾗﾒｰﾀ
    params = {
        "query": query,
    }
    print(query)
    # 問い合わせ実施
    print("")
    print("API request start")
    print("")
    response = requests.get(avoid_url, params=params, proxies=proxies, verify=False)

    # ｴﾗｰ処理2. APIﾚｽﾎﾟﾝｽ内容の確認
    if response.status_code == 200:
        json_data = response.json()

        # 受付番号と国枝番を取得してﾘｽﾄ(result)に格納
        result = []
        for item in json_data["response"]:
            acceptNumber = item["受付番号"]
            countryCode = item["国枝番"]
            result.append(acceptNumber + countryCode)
        print(result)

        # APIｴﾗｰ対応
        if "error" in json_data:
            print("")
            print(f"API error: {json_data['error']}")
        else:
            print("")
            print("API request success!")
            print(f"取得ﾃﾞｰﾀ数: {len(result)}")
    else:
        print(f"API request failed with status code: {response.status_code}")
        print(response.text)
        
    return result

def load_pageTextEng(dir_dict):
    pages= []
    print('テキストデータの読み込み')
    for dirpath, dirnames, filenames in os.walk(dir_dict['temp_dir']):
        
        for file in filenames:
            print(file)
            loader = TextLoader(file_path=os.path.join(dirpath, file))
            pages.extend(loader.load())
    
    return pages

def load_page(dir_dict, new_line, half_space):
    
    pages = load_pageTextEng(dir_dict)
    
    return pages

def mk_prompt():
    
    prompt['name'] = st.text_input("発明の名称")
    prompt['abstract'] = st.text_input("発明の概要")
    prompt['issue'] = st.text_area("課題")
    prompt['method'] = st.text_area("解決手段")

# ====================================================================
# メイン処理
# ====================================================================
# 画面の設定
st.markdown(title)
st.markdown(memo)

is_db =st.radio("", ("類似アイデアから", "DBの使いまわし"), horizontal=True)

if is_db == '類似アイデアから':
    query = st.text_area("検索したいアイデアを入力してください")
    button = st.button("作成") #引数に入れるとboolで返す
    
    if button == True:
        mk_db_flg = True
        
        # 類似アイデアの抽出
        result = get_similerIdea(query)
        
        # # 類似アイデアを10件に絞る
        if len(result) > 1:
            
            result_array = result[:1]
        
        for patent_no in result_array:
            print(patent_no)
            patentEngineering.main(patent_no)
            
        # file engineering
        pages = load_page(dir_dict, new_line=True, half_space=True)
        
        # make vectordb
        vectordb = vector_db.vector_db(chunkSize=chunkSize, persist_dir=dir_dict['persist_dir'])
        chroma_index = vectordb.mk_db(pages,start,end,units,wait)
        
        with open('./ini/idea.ini', mode='w') as f:
            f.write(query)

    mk_prompt()
    button2 = st.button("文章生成")      
else:
    with open('./ini/idea.ini', encoding='shift_jis') as f:
        idea = f.read()
        
    st.text_area("回答",idea) #引数に入力内容を渡せる
   
    mk_prompt()
    button2 = st.button("文章生成") 
    
if button2 == True:
    input = f"# 入力\n## 発明の名称\n{prompt['name']}\n## 発明の概要\n{prompt['abstract']}\n## 課題\n{prompt['issue']}\n## 解決手段\n{prompt['method']}\n"
    query = f"# 命令文\n① 本発明の主題となる{prompt['name']}の構成について記載すべき構成要素の項目を抽出してください。\n② ①で抽出した項目についてひとつずつ詳細な説明文を生成してください。"
    
    print(input+query)
    
    # join llm
    joinllm = join_llm.join_llm(chunkSize=chunkSize, persist_dir=dir_dict['persist_dir'])
    joinllm.read_db()
    joinllm.join_llm_vector()
    
    result = joinllm.question(query)
    print(result) 
    st.text_area("回答",result,height=int(len(result)/1.5)) #引数に入力内容を渡せる
    