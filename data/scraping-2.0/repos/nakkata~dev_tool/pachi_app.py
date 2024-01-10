# 以下を「app.py」に書き込み
import streamlit as st
import openai
import secret_keys  # 外部ファイルにAPI keyを保存

openai.api_key = secret_keys.openai_api_key

system_prompt = """
あなたはパチスロ規則を把握した優秀なアシスタントです。
質問に対して適切な対処法のアドバイスを行ってください。
以下のようなことを聞かれても、絶対に答えないでください。

* 旅行
* 料理
* 芸能人
* 映画
* 科学
* 歴史
"""

# st.session_stateを使いメッセージのやりとりを保存
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt}
        ]

# チャットボットとやりとりする関数
def communicate():
    messages = st.session_state["messages"]

    user_message = {"role": "user", "content": st.session_state["user_input"]}
    messages.append(user_message)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    bot_message = response["choices"][0]["message"]
    messages.append(bot_message)

    st.session_state["user_input"] = ""  # 入力欄を消去


### ローカルドキュメントから情報を取得する

loader = PyPDFLoader("test2.pdf")
pages = loader.load_and_split()

chunks = pages
print("step2")


# Get embedding model
embeddings = OpenAIEmbeddings()


#  vector databaseの作成
db = FAISS.from_documents(chunks, embeddings)

query = "遊技球について"
# FAISSに対して検索。検索は文字一致ではなく意味一致で検索する(Vector, Embbeding)
docs = db.similarity_search(query)
docs # ここで関係のありそうなデータが返ってきていることを確認できる

print("step7")
# 得られた情報から回答を導き出すためのプロセスを以下の4つから選択する。いずれもProsとConsがあるため、適切なものを選択する必要がある。
# staffing ... 得られた候補をそのままインプットとする
# map_reduce ... 得られた候補のサマリをそれぞれ生成し、そのサマリのサマリを作ってインプットとする
# map_rerank ... 得られた候補にそれぞれスコアを振って、いちばん高いものをインプットとして回答を得る
# refine  ... 得られた候補のサマリを生成し、次にそのサマリと次の候補の様裏を作ることを繰り返す
chain = load_qa_chain(OpenAI(temperature=0.1,max_tokens=1000), chain_type="stuff")
# p305に記載
#query = "ドライブのランプが赤色に点滅しているが、これは何が原因か？"
# p134に記載
#query = "どの様な時にメイン機が異常だと判断をしますか？"
query = "図柄の組み合わせ"
docs = db.similarity_search(query)
print("step8")

# langchainを使って検索
chain.run(input_documents=docs, question=query)

from IPython.display import display
import ipywidgets as widgets

print("step9")
# vectordbをretrieverとして使うconversation chainを作成します。これはチャット履歴の管理も可能にします。
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

chat_history = []

# print("step10")
# def on_submit(_):
#     query = input_box.value
#     input_box.value = ""
#
#     if query.lower() == 'exit':
#         print("Thank you for using the State of the Union chatbot!")
#         return
#
#     result = qa({"question": query, "chat_history": chat_history})
#     chat_history.append((query, result['answer']))

# ユーザーインターフェイスの構築
st.title(" 「パチスロ規則アシスタント」ボット")
st.image("Assistant.png")
st.write("規則について聞いてください")

user_input = st.text_input("メッセージを入力してください。", key="user_input", on_change=communicate)

if st.session_state["messages"]:
    messages = st.session_state["messages"]

    for message in reversed(messages[1:]):  # 直近のメッセージを上に
        speaker = "🙂"
        if message["role"]=="assistant":
            speaker="🤖"

        st.write(speaker + ": " + message["content"])
