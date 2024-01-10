import streamlit as st
import openai

#ChatGPT에게 글 요약을 요청하는 함수
def askGPT(prompt, apiKey):
    client = openai.OpenAI(api_key=apiKey)
    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {"role":"user", "content":prompt}
        ]
    )
    finalResponse = response.choices[0].message.content
    return finalResponse

## main함수
def main():
    st.set_page_config(page_title="요약 프로그램")

    #session_state 초기화
    if "OPENAI_API" not in st.session_state: #not in : not in 연산자는 해당 값이 리스트에 없는지 확인하는 연산자
        st.session_state["OPENAI_API"] = "" #session_state : session_state는 사용자가 웹 페이지를 사용하는 동안에만 유지되는 상태를 저장하는 객체
    
    with st.sidebar:
        open_apiKey = st.text_input(label = 'OpenAI API Key',placeholder= 'Enter your api key')

        if open_apiKey:
            st.session_state["OPENAI_API"] = open_apiKey #apikey를 session_state에 저장
        st.markdown('---')

    st.header(":memo: 요약 프로그램")
    st.markdown('---')

    text = st.text_area('요약할 글을 입력하세요.')
    if st.button("요약"):
        prompt = f'''
        **Instructions** :
    - You are an expert assistant that summarizes text into **Korean language**.
    - Your task is to summarize the **text** sentences in **Korean language**.
    - Your summaries should include the following :
        - Omit duplicate content, but increase the summary weight of duplicate content.
        - Summarize by emphasizing concepts and arguments rather than case evidence.
        - Summarize in 3 lines.
        - Use the format of a bullet point.
    -text : {text}
    '''
            
        st.info(askGPT(prompt,st.session_state["OPENAI_API"]))

if __name__ == '__main__': #__name__
    main()  # main 함수 실행

