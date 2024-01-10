import openai
import streamlit as st
from streamlit_feedback import streamlit_feedback

openai.api_base = "https://oai.langcore.org/v1"

if "mail" not in st.session_state:
    st.session_state["mail"] = None
if "prompt" not in st.session_state:
    st.session_state["prompt"] = None
if "request_body" not in st.session_state:
    st.session_state["request_body"] = None
if "response_body" not in st.session_state:
    st.session_state["response_body"] = None


def main():
    st.title("メール自動作成デモ")
    st.write("URLを入力すると、その内容を元にChatGPTを使ってメール文面を自動作成するデモ")

    url = st.text_input(
        "参照先URL",
        "https://toyota-career.snar.jp/jobboard/detail.aspx?id=Vx6tWwR9tzJH6UJagFspxw",
    )
    mail_template = st.text_area("作成するメールのテンプレート", get_mail_template(), height=500)

    if st.button("メールを作成する"):
        with st.spinner("メールを作成中です..."):
            create_mail(url, mail_template)

    if st.session_state["mail"] != None and st.session_state["prompt"] != None:
        mail = st.session_state["mail"]
        prompt = st.session_state["prompt"]
        request_body = st.session_state["request_body"]
        response_body = st.session_state["response_body"]

        st.markdown(
            '<span style="font-size:0.8em;color:gray">メールを作成しました！</span>',
            unsafe_allow_html=True,
        )
        st.text_area("作成されたメール", mail, height=500)

        streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="フィードバックをお願いします",
            on_submit=on_submit,
            args=[request_body, response_body, st.secrets["OPENAI_API_KEY"]],
        )

        expander = st.expander("実行したプロンプト", expanded=False)
        with expander:
            st.text(prompt)


def create_mail(url, mail_template):
    from trafilatura import fetch_url, extract
    from trafilatura.settings import use_config

    config = use_config()
    config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "1000")
    downloaded = fetch_url(url)
    result = extract(downloaded, config=config)

    # テキストが長すぎる場合は、一部を削除します。
    content = result
    if len(content) > 1000:
        content = result[:1000]

    prompt = f"""
    企業情報 {{
    {content}
    }}

    MAIL_TEMPLATE{{
    {mail_template}
    }}

    制約条件
    - 企業情報を見て、MAIL_TEMPLATEにある[]を全て埋めてください
    - MAIL_TEMPLATE:の文章をそのまま使ってください
    - []は削除してください
    - []を埋められない場合は削除してください

    補完したMAIL_TEMPLATE:
    """
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": prompt},
        ],
        "user": "山田太郎",
    }
    res = openai.ChatCompletion.create(**request_body)
    mail = res.choices[0].message.content

    st.session_state["request_body"] = request_body
    st.session_state["response_body"] = res
    st.session_state["mail"] = mail
    st.session_state["prompt"] = prompt

    return mail, prompt


def get_mail_template():
    day1, day2, day3, day1_youbi, day2_youbi, day3_youbi = get_jikoku()

    MAIL_TEMPLATE = f"""
[企業名]様

初めまして、田中太郎と申します。

ホームページを拝見し、[企業の困っていること]で課題を抱えられているのではないかと思い、ご連絡させていただきました。

私は[企業の困っている領域]での経験があります。
[企業に刺さりそうな謳い文句]

ご多用かと存じますが、下記の中から30分、面接のお時間を頂戴できますと幸いです。

- {day1} 11:00 ~ 18:00
- {day2} 11:00 ~ 18:00
- {day3} 11:00 ~ 18:00

ご連絡を心よりお待ち申し上げております。
    """
    return MAIL_TEMPLATE


def get_jikoku():
    import datetime
    import workdays
    import locale

    locale.setlocale(locale.LC_TIME, "")
    today = datetime.date.today()
    day1 = workdays.workday(today, days=2)
    day2 = workdays.workday(today, days=3)
    day3 = workdays.workday(today, days=4)
    day1_youbi = day1.strftime("%a")
    day2_youbi = day2.strftime("%a")
    day3_youbi = day3.strftime("%a")
    day1 = day1.strftime("%-m/%-d")
    day2 = day2.strftime("%-m/%-d")
    day3 = day3.strftime("%-m/%-d")
    return day1, day2, day3, day1_youbi, day2_youbi, day3_youbi


def on_submit(feedback, request_body, response_body, openai_api_key):
    import requests
    import json

    feedback_type = feedback["type"]
    score = feedback["score"]
    if score == "👍":
        score = 1
    elif score == "👎":
        score = 0
    optional_text_label = feedback["text"]

    url = "http://langcore.org/api/feedback"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai_api_key,
    }
    data = {
        "request_body": request_body,
        "response_body": response_body,
        "feedback_type": feedback_type,
        "score": score,
        "optional_text_label": optional_text_label,
    }
    requests.post(url, headers=headers, data=json.dumps(data))
    # st.toast("フィードバックを送信しました。")　バージョン違いでなぜか表示されない
    # URLを表示する
    st.write("フィードバックはこちらに記録されます: https://langcore.org/feedback")


if __name__ == "__main__":
    main()
