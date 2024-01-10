import streamlit as st
from openai import OpenAI

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

st.title("✍️ AI_카피라이터")
st.subheader("AI를 이용하여 손쉽게 마케팅 문구를 생성해보세요.")

auto_complete = st.toggle(label="예시로 채우기")
example = {
    "name": "카누",
    "desc": "집에서도 카페 맛을 낼 수 있는 커피믹스",
    "keywords": ["브라질", "원두", "풍미"]
}


def generate_prompt(product_name, product_desc, num, max_length, keywords):
    prompt = f"""
제품 혹은 브랜드를 SNS에 광고하기 위한 문구를 {num}개 생성해주세요.
반드시 {max_length} 단어 이내로 생성해주세요.
자극적이고 창의적으로 작성하세요.
명사 위주로 간결하게 작성해주세요.
이모지를 적절하게 섞어주세요.
키워드가 주어질 경우, 반드시 키워드 중 하나를 포함해야합니다.
---
제품명: {product_name}
제품설명: {product_desc}
키워드: {keywords}
---
    """.strip()
    return prompt


def request_chat_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 전문 카피라이터입니다."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    return response


def print_streaming_response(response):
    message = ""
    placeholder = st.empty()
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            message += delta.content
            placeholder.markdown(message + "▌")
    placeholder.markdown(message)


with st.form("form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input(
            "제품/브랜드 이름(필수)",
            value=example["name"] if auto_complete else "",
            placeholder=example["name"]
        )
    with col2:
        max_length = st.number_input(
            label="최대 단어 수",
            min_value=5,
            max_value=20,
            step=1,
            value=10
        )
    with col3:
        num = st.number_input(
            label="생성할 문구 수",
            min_value=1,
            max_value=10,
            step=1,
            value=5
        )
    desc = st.text_input(
        label="제품 간단 정보 (필수)",
        value=example["desc"] if auto_complete else "",
        placeholder=example["desc"]
    )
    st.text("포함할 키워드(최대 3개까지 허용)")
    col1, col2, col3 = st.columns(3)
    with col1:
        keyword_one = st.text_input(
            label="keyword_1",
            label_visibility="collapsed",
            value=example["keywords"][0] if auto_complete else "",
            placeholder=example["keywords"][0]
        )
    with col2:
        keyword_two = st.text_input(
            label="keyword_2",
            label_visibility="collapsed",
            value=example["keywords"][1] if auto_complete else "",
            placeholder=example["keywords"][1]

        )
    with col3:
        keyword_three = st.text_input(
            label="keyword_3",
            label_visibility="collapsed",
            value=example["keywords"][2] if auto_complete else "",
            placeholder=example["keywords"][2]
        )
    submit = st.form_submit_button("제출하기")
if submit:
    if not name:
        st.error("제품/브랜드 이름을 입력해주세요.")
    elif not desc:
        st.error("제품 간단 정보를 입력해주세요.")
    else:
        keywords = [keyword_one, keyword_two, keyword_three]
        keywords = [x for x in keywords if x]
        prompt = generate_prompt(
            product_name=name,
            product_desc=desc,
            num=num,
            max_length=max_length,
            keywords=keywords
        )
        response = request_chat_completion(prompt)
        print_streaming_response(response)
