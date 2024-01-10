import openai
import streamlit as st
from openai import OpenAI
from pathlib import Path


st.title("ファイルを読み込み、ファイルを出力")

client = OpenAI()

if st.button("ファイルを作成する"):
    instruction = """
# タスクの目的
エクセル形式で20日分の献立表を出力する。

# 献立の条件
総カロリーは一食あたり400カロリー前後とする。
20日間の中で同じ品目を繰り返さない。
一食内での品目カテゴリのバランスを考慮する。副菜は重複しても良い。
1日(食)あたりの品目数は、総カロリーが400前後となるように4〜6品目をピックアップする。

# 献立の構成
主食: 1品目
主菜: 1品目
副菜: 2品目
汁物: 1品目
デザート: 1品目

# データの関連性
品目に使用される食材の栄養価は、食材表に記載の1gあたりの栄養価情報を基に、食材の使用量で掛け合わせて計算する。
品目IDを使用して、品目名とのマッピングを行う。
品目に使用される食材は、品目_食材_調味料表を参照する。

# 出力する献立表のカラム
day: 日数情報
献立: 1日(食)あたりに使用している品目名を記載
赤、黄、緑、調味料: 品目に使用している食材のうち、各食材カテゴリに該当する食材を全て表示
一食あたりのカロリー
一食あたりのたんぱく質(g)
一食あたりの脂質(g)
    """
    # アシスタントを作成
    file = client.files.create(
        file=Path("/Users/mugiro/genai-poc/assistant/data/input.xlsx"),
        purpose='assistants'
    )
    st.write("ファイルアップロード完了")

    assistant = client.beta.assistants.create(
        name="cook",
        instructions=instruction,
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-1106-preview",
        file_ids=[file.id]
    )
    st.write("アシスタントを新規作成")
    st.write(assistant)

    # スレッドを作成
    thread = client.beta.threads.create(
        messages=[
            {
            "role": "user",
            "content": "条件に従った献立表のexcelファイルを作成してください。"
            }
        ]
)
    st.write("スレッドを新規作成")
    st.write(thread)

    # スレッドを実行
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="see the attached file and get the answer file"
    )
    st.write("スレッドを実行")
    st.write(run.status)
    
    ## まつ
    import time
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        st.write(run.status)
        time.sleep(1)


    # メッセージを取得
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    st.write("メッセージを取得")
    st.write(messages.data[0].content[0].text.value)


    # アシスタントの削除
    client.beta.assistants.delete(
        assistant_id=assistant.id
    )