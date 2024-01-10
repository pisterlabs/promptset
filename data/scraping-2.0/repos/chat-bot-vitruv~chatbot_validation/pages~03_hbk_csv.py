import streamlit as st
import pandas as pd
import csv
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.few_shot import FewShotPromptTemplate

from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore

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
st.title('HBK 금쪽이 CSV File Donwloader')


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
        1. qusetionId | test | expText | answer 로 이루어진 csv 파일을 넣는다.
        2. 사이드바의 버튼을 클릭한다.
        3. 결과 csv 파일이 생성되길 기다린다.
            생각보다 생성이 느립니다 30문제 15분?
        4. 결과 다운로드 버튼을 클릭한다.
        """
    )
        
    # 사이드바 생성
    sidebar = st.sidebar
    uploaded_file= None;
    df = None;
        
    with st.sidebar:
        uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        
    if df is not None:
        st.write(df)

    # 사이드바에 버튼 추가
    if sidebar.button('버튼 클릭'):
        status_placeholder = st.empty()  # 상태 메시지를 위한 임시 홀더

        for index, row in df.iterrows():
            response = None;
            question = row["test"]
            solving = row["expText"]

            status_placeholder.write(f"{index}번째 해설을 생성하는 중...")

            response = chain.invoke({"question": question, "solving": solving})

            if response:
                df.at[index, 'response'] = response.content


        csv_result = df.to_csv(index=False).encode('utf-8-sig')

        st.download_button(label="결과 다운로드", data=csv_result, file_name='processed_results.csv', mime='text/csv')        



if __name__ == '__main__':
    main()
