import openai
import streamlit as st
from bardapi import Bard

#ChatGPT 함수
def askGPT(prompt, apiKey):
    client = openai.OpenAI(api_key=apiKey)
    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {"role":"user", "content":prompt}
        ]
    )
    return response.choices[0].message.content, \
            response.usage.completion_tokens, \
            response.usage.prompt_tokens 
    # 메시지 , 응답 토큰 수, 프롬프트 토큰 수/ 리턴이 3개이다. 이 함수 앞에는 변수가 3개여야함

#바드 함수
def askBard(prompt):
    bard = Bard(token=st.session_state["BARD_TK"],timeout=120) #timeout은 120초
    result = bard.get_answer(prompt) #.get_answer()는 질문을 입력받아 답변을 반환하는 함수
    return result["choices"][0]["content"], result["choices"][1]["content"], result["choices"][2]["content"] #result는 딕셔너리 형태이므로, 딕셔너리의 키값을 이용하여 value를 반환한다.

#main (스트림릿을 위한 함수)
def main():
    st.set_page_config( 
        page_title="ChatGPT vs Bard", 
        page_icon="🤖", 
        layout="wide"
    )
    
    st.title(":right-facing_fist: ChatGPT vs Bard 비교 프로그램 :left-facing_fist:")
    st.markdown('---')

     #session_state 초기화
    if "model" not in st.session_state: #모델 저장
        st.session_state["model"] = "" #모델이 없다면 만들어서 넣어라.
    if "OPENAI_API" not in st.session_state: #openai api 저장
        st.session_state["OPENAI_API"] = "" 
    if "BARD_TK" not in st.session_state: #bard api 저장
        st.session_state["BARD_TK"] = "" 
   
    with st.sidebar: #아래 코드들은 sidebar에 위치 (with st.sidebar는 함수바디)
        open_apiKey = st.text_input(label = 'OpenAI API Key',placeholder= 'Enter your api key', value='', type='password') #placeholder는 힌트, value는 기본값, type은 입력값의 타입

        if open_apiKey: #openai api key 입력시(있다면)
            st.session_state["OPENAI_API"] = open_apiKey
            openai.api_key = open_apiKey

        st.radio(label = "Select Model", options = ["gpt-4.0", "gpt-3.5-turbo", "gpt-3.5"])
        st.markdown('---')

        bard_token = st.text_input(label = 'Bard Token',placeholder= 'Enter your Bard token', value='', type='password')

        if bard_token: #bard api key 입력시(있다면)
            st.session_state["BARD_TK"] = bard_token
        st.markdown('---')

    st.header("프롬프트를 입력하세요.")
    prompt = st.text_input(" ")
    st.markdown('---')

    col1, col2 = st.columns(2)
    with col1:
         st.header("ChatGPT")
         if prompt:
            if st.session_state["OPENAI_API"]:
                result, completion_token, prompt_token = \
                    askGPT(prompt, st.session_state["OPENAI_API"])
                st.markdown(result)
                if st.session_state["model"] == "gpt-4.0":
                    completion_bill = completion_token * 0.3
                    prompt_bill = prompt_token * 0.6
                    total_bill = (completion_bill + prompt_bill) * 0.001
                elif st.session_state["model"] == "gpt-3.5-turbo":
                    completion_bill = completion_token * 0.02
                    prompt_bill = prompt_token * 0.015
                    total_bill = (completion_bill + prompt_bill) * 0.001
                else :
                    completion_bill = completion_token * 0.06
                    prompt_bill = prompt_token * 0.03
                    total_bill = (completion_bill + prompt_bill) * 0.001
            else:
                st.info("OpenAI API Key를 입력하세요.")

    with col2:
        st.header("Bard")
        if prompt:
            if st.session_state["BARD_TK"]:
                result1, result2, result3 = askBard(prompt)
                st.markdown('### 답변1')
                st.markdown(result1)
                st.markdown('### 답변2')
                st.markdown(result2)
                st.markdown('### 답변3')
                st.markdown(result3)
                total_bill = 0
            else:
                st.info("Bard Token Key를 입력하세요.")





if __name__ == "__main__":
    main()

    