import openai
import streamlit as st

# OpenAI API key를 설정합니다.
openai.api_key = "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"

# Streamlit 페이지의 설정을 변경합니다.
st.set_page_config(
    page_title="제안서 작성기",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto",
)

# 사이드바
st.sidebar.header("작성 방법")
st.sidebar.markdown("1. 오른쪽의 텍스트 박스에 제안서 작성에 필요한 정보를 입력하세요.")
st.sidebar.markdown("2. '제안서 작성' 버튼을 클릭하세요.")
st.sidebar.markdown("3. 오른쪽에 제안서가 생성됩니다.")
st.sidebar.markdown("---")

# 제목, 회사명, 제안서의 내용 등 필요한 정보를 입력받는 창을 만듭니다.
st.title("제안서 작성기")
project_title = st.text_input("프로젝트 제목을 입력하세요.")
company_name = st.text_input("회사명을 입력하세요.")
project_summary = st.text_area("프로젝트 요약을 입력하세요.", height=200)
project_description = st.text_area("프로젝트 설명을 입력하세요.", height=400)

# 입력받은 정보를 OpenAI API를 활용해 제안서의 내용을 생성합니다.
if st.button("제안서 작성하기"):
    with st.spinner("제안서 작성 중입니다. 잠시만 기다려주세요..."):
        prompt = (f"제목: {project_title}\n"
                  f"회사명: {company_name}\n"
                  f"요약: {project_summary}\n"
                  f"설명: {project_description}\n"
                  "제안서 작성을 시작합니다.\n\n"
                  "다음은 우리 회사가 이 프로젝트를 수행하기에 완벽하게 적합한 이유입니다:\n\n")
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.7,
        )
        proposal = response.choices[0].text

    # 제안서 내용을 출력합니다.
    st.subheader("제안서 내용")
    st.write(proposal)
