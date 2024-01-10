import os
import pickle
from langchain.chat_models import ChatOpenAI
import openai
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from typing import Any, Dict, List
import streamlit as st
import openai
import json

st.components.v1.html("""
<style>
[data-testid="stVerticalBlock"] > [data-stale="false"] > [class="stMarkdown"][style*="width:"] >[data-testid="stMarkdownContainer"] p {
    position: fixed;
    font-family: "Rounded M+ 1c medium";
    font-size: 1.5vmax;
    color: white;
    justify-content: center;
    align-items: center;
    bottom: 0%;
    left: 0%;
    padding: 1% 5% 1% 5%;
    width: 90%;
    height: 30%;
    animation: fadeOut 4s;
    overflow-y: auto;
}
</style>
""", height=0)

class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """ Copied only streaming part from StreamlitCallbackHandler """
    
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        
        token = token.replace('\n\n', '\n')
        token = token.replace('\n', '  \n')
        
        if token==' ':
            print('space')
        if token=='\n':
            print('n')
        # if token=='\n\n':
            # print('nn')

        if len(token)>1:
            for s in token:
                self.tokens_stream += s
                self.tokens_area.markdown(self.tokens_stream)
        else:
            self.tokens_stream += token
            self.tokens_area.markdown(self.tokens_stream)
            


# 定数の設定
openai.api_key = os.environ.get('OPEN_AI_KEY')

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    # model_name="gpt-4",
    temperature=0.2,
    streaming=True

)

system_message = SystemMessage(
        content="あなたは進路指導のプロです。高校生の進路指導を担当しています。傾聴と肯定を大切にするキャリアコンサルタントです。１年生から３年生の進路に対するアドバイスをしています。生徒に対して、自分らしく納得の行くキャリアに進むために必要なことを教えています。相手の名前を「さん」付けて呼びましょう。挨拶は省略してください。絵文字を多用してください。"
    )

def response_logic1(name, answers):
    message = [
        system_message,
        HumanMessage(
            content=f"私の名前は{name}です。私は高校{answers[0]}です。進路の状況は「{answers[1]}」です。200文字程度でアドバイスください"
        ),
    ]
    return llm(message, callbacks=[SimpleStreamlitCallbackHandler()]).content


def response_logic2(name,answers):
    message = [
        system_message,
        HumanMessage(
            content=f"私の名前は{name}です。私は本を読むのが{answers[0]}。本を読む頻度は「{answers[1]}」です。200文字程度でアドバイスください."
        ),
    ]
    return llm(message, callbacks=[SimpleStreamlitCallbackHandler()]).content


    
def response_logic3(name, answers):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        # model_name="gpt-4",
        temperature=0.2,
        streaming=True
    )

    
    with open('static/denen_book_v2.pkl', 'rb') as f:
        db = pickle.load(f)  # 復元

    functions = [
            {
                "name": "i_am_json",
                "description": "抽出された特徴を JSON として処理します。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_keyword": {
                            "type": "string",
                            "description": "新キーワード",
                        },
                    #     "reason": {
                    #         "type": "string",
                    #         "description": "変換理由",
                    # },
                },
            }
            }
    ]

    
    prompt_template1 = f"""
# 指示
以下のキーワードから連想される意外性のあるキーワードに変換してください。
    
# キーワード
{answers[0]}

# 変換例
## 「物理学」 →　「ロマン」
「物理」が好きな人は宇宙が好きで「ロマン」という言葉が好きかもしれない
## 「楽したい」 →　「FIRE」
「楽したい」人はアーリーリタイアしたくて、「FIRE」に興味があるはずだ
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.5,
        messages=[
            {"role": "user", "content": prompt_template1}],
        functions=functions,
        function_call={"name": "i_am_json"},
    )
    
    message = response["choices"][0]["message"]
    new_keyword = json.loads(message["function_call"]["arguments"])['new_keyword']
    # keyword = answers[0] + "、" +  new_keyword['new_keyword']
    # keyword = answers[0] + "、" +  new_keyword['new_keyword']
    print(new_keyword)
    keyword = answers[0]

    docs = db.similarity_search_with_relevance_scores(f'「{answers[0]}」', k=8)
    outputs = []
    for doc, score in docs:
        book_name = doc.page_content
        author = doc.metadata.get('著者', '不明')
        publisher = doc.metadata.get('出版社', '不明')
        location = doc.metadata.get('請求記号', '不明')
        output = f'書籍名「{book_name}」\n著者:{author}、出版社:{publisher}、配架場所:{location}\n'
        outputs.append(output)
    
    docs = db.similarity_search_with_relevance_scores(f'「{new_keyword}」', k=4)
    for doc, score in docs:
        book_name = doc.page_content
        author = doc.metadata.get('著者', '不明')
        publisher = doc.metadata.get('出版社', '不明')
        location = doc.metadata.get('請求記号', '不明')
        output = f'書籍名「{book_name}」\n著者:{author}、出版社:{publisher}、配架場所:{location}\n'
        outputs.append(output)
        
    context = '\n,'.join(outputs)
    
    prompt_template = """
貴方は図書館の司書です。
以下の書籍リストから、ユーザのキーワードに近い書籍を{n}冊必ず選んでください。
適当なものがなくても、必ず{n}冊答えてください。
おすすめ理由では「{name}さん」と、呼んであげましょう。

{context}

キーワード: {question}
意外なキーワード: {new_keyword}

Answer in Japanese:
あなたのキーワード: {question}
ーーーーーーーーーーーーーーーー
📚書籍名1：「book_name」
✒著者：「author_name」、🏢出版社:「publisher」, 🔎配架場所：「book_place」
💡おすすめ理由： reason
ーーーーーーーーーーーーーーーー
📚書籍名2:「book_name」
✒著者：「author_name」、🏢出版社:「publisher」, 🔎配架場所：「book_place」
💡おすすめ理由： reason
ーーーーーーーーーーーーーーーー
意外なキーワード: {new_keyword}
📚書籍名3:「book_name」
✒著者：「author_name」、🏢出版社:「publisher」, 🔎配架場所：「book_place」
💡おすすめ理由： reason
    """
    # prompt_template = prompt_template.format(n=3, name=name, context=context, question="{question}")
    prompt_template = prompt_template.format(n=3, name=name, context=context, question=keyword, new_keyword=new_keyword)
    
    message = [
        HumanMessage(
            content = prompt_template
        ),
    ]
    return llm(message, callbacks=[SimpleStreamlitCallbackHandler()]).content



def response_logic4(name, answers):

    prompt_template = """
# 指示内容
キーワードから将来考えられる仕事を５つ紹介してください。業種はバラバラにしてください。

# 詳細条件
最初の３つは既存にある仕事を。４つ目は、最先端の技術を使い、100年後これから生まれるような仕事を。最後の一つはキーワードとは全く関係ないけど、現代の社会問題の課題を解決できる意外性のある仕事を紹介してください。

# キーワード
{keyword}

# 出力形式
ーーーーーーーーーーーーーーーー
あなたのキーワード: {keyword}
ーーーーーーーーーーーーーーーー
💼職業①「職業名」
💡【業種】：XXX
💭【概要】：XXX
ーーーーーーーーーーーーーーーー
💼職業②「職業名」
💡【業種】：XXX
💭【概要】：XXX
ーーーーーーーーーーーーーーーー
"""

    message = [
        system_message,
        HumanMessage(
            content = prompt_template.format(keyword=answers[0])
        ),
    ]
    return llm(message, callbacks=[SimpleStreamlitCallbackHandler()]).content
    
