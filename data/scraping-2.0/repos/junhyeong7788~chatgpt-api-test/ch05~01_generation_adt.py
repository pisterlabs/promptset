import streamlit as st
import openai

#ChatGPT에게 요청하는 함수
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

def main():
    st.set_page_config(page_title="광고문구생성프로그램")

    #session_state 초기화
    if "OPENAI_API" not in st.session_state: 
        st.session_state["OPENAI_API"] = "" 
    
    with st.sidebar:
        open_apiKey = st.text_input(label = 'OpenAI API Key',placeholder= 'Enter your api key')

        if open_apiKey:
            st.session_state["OPENAI_API"] = open_apiKey 
        st.markdown('---')

    st.header(":speech_balloon: 광고 문구 생성 프로그램")
    st.markdown('---')

    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("제품명", placeholder="제품명을 입력") #입력받고 변수에 저장
        strength = st.text_input("제품 특징", placeholder="제품특징을 입력")
        keywords = st.text_input("필수 포함 키워드", placeholder="필수 포함 키워드를 입력")

    with col2:
        brand = st.text_input("브랜드 명", placeholder="Apple, Google, Microsoft 등")
        toneManner = st.text_input("톤 앤 매너", placeholder="발랄하게, 유머러스하게, 감성적으로 등")
        value = st.text_input("브랜드 핵심 가치", placeholder="필요시 작성")

    if st.button("광고 문구 생성"):
        prompt = f'''
        아래 내용을 참고해서 1~2줄짜리 광고문구 5개 작성해줘.
        - 제품명 : {name}
        - 브랜드 명 : {brand}
        - 브랜드 핵심 가치 : {value}
        - 제품 특징 : {strength}
        - 필수 포함 키워드 : {keywords}
        - 톤 앤 매너 : {toneManner}    
        '''
            
        st.info(askGPT(prompt,st.session_state["OPENAI_API"]))

if __name__ == "__main__":
    main()