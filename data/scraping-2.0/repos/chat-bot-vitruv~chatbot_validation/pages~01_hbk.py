import streamlit as st
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.few_shot import FewShotPromptTemplate

from langchain.prompts.prompt import PromptTemplate
from files.example_data import examples
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'NanumBarunGothic'

def replace_text(input_text):
    # img 태그 제거
    input_text = re.sub(r'<img[^>]*>', '', input_text)

    # <br> 태그 제거
    input_text = input_text.replace('<br>', '')

    # <b> 태그 제거 (여는 태그와 닫는 태그 모두 제거)
    input_text = re.sub(r'</?b>', '', input_text)

    # [ ] -> $ $ 로 변경
    input_text = input_text.replace('[', '$').replace(']', '$')

    return input_text

def render_latex_for_streamlit(text):
    """
    주어진 텍스트 내의 LaTeX 수식을 포함하여 Streamlit에 맞게 렌더링하는 함수입니다.

    :param text: 처리할 전체 텍스트
    """
    # align* 환경이 시작되었는지 추적
    in_align_block = False
    align_block = ""

    lines = text.split('\n')
    for line in lines:
        if r"\begin{align*}" in line:
            in_align_block = True
            align_block += line + '\n'
        elif r"\end{align*}" in line:
            in_align_block = False
            align_block += line
            st.latex(align_block)
            align_block = ""
        elif in_align_block:
            align_block += line + '\n'
        else:
            st.write(line)


# 애플리케이션의 제목 설정
st.title('HBK 금쪽이')


llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    temperature=0.1,
    streaming=True,
    verbose="true",
    
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

example_prompt = PromptTemplate.from_template("Human: {question} {solving} {answer}\n {aiAnswer}")


prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="""
    Human :
        Question : {question}
        Solving : {solving}
    """,
    prefix="""
        You are a competent math teacher. Students love you, and you love them, too. You give them explanations on math problems with affection.
Explain it easily even for students who are not good at math.
The commentary is explained in detail step by step, and at the end of each step, words that induce students to solve problems such as 'Shall we do this?' are added.
The commentary should not be answered, and the commentary should never be told wrong. It should also deliver accurate commentary to students. There should be no speculative words.
You must answer in Korean.
    """,
    input_variables=["question", "solving"],
)


chain = prompt | llm

def main():
    st.markdown(
        """
        ### 사용 방법
        1. [어드민 페이지](https://ai.matamath.net/admin/question)에서 원하는 문제를 선택한 후 문제의 개념/용어 태깅을 off한다
        2. 문제와 텍스트를 입력란에 복붙한다
        3. 답안 생성 타입을 선택한다
        4. 실행 버튼 클릭

        > 답안생성까지 최대 2분 정도 소요됩니다.
            
        """
    )
        
    # 사이드바 생성
    sidebar = st.sidebar
    sidebar.header('문제/해설 입력')

    # 사이드바에 텍스트 입력 필드 추가
    question_input = sidebar.text_area('문제 입력')
    answer_input = sidebar.text_area('해설 입력')

    # HTML/CSS를 사용하여 테두리가 있는 텍스트 박스 생성
    st.markdown('<style>div.stTextInput>div{border:2px solid #4CAF50;}</style>', unsafe_allow_html=True)

    # 상단의 단일 텍스트 박스 추가
    # text_box_1 = st.text_area("Prompt 입력 데이터(전처리", replace_text(question_input), height=100)

    # 사이드바에 버튼 추가
    if sidebar.button('버튼 클릭'):
        response = None;

        with st.status("해설을 생성하는 중...") as status:
            response = chain.invoke({"question" : question_input, "solving" : answer_input})

        if response:
            st.markdown("---")
            st.markdown("## Raw Result Data")
            st.write(response.content)
            st.markdown("---")
            st.markdown("## st.latex Result Data")
            render_latex_for_streamlit(response.content)

        



if __name__ == '__main__':
    main()
