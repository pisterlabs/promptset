import openai
import streamlit as st
from googletrans import Translator
import requests
import deepl

#GPT
def tranGPT(text, apiKey):
    client = openai.OpenAI(api_key=apiKey)
    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {"role":"system", "content":"Translate the following text into Korea. Text to translate : {text}".format(text=text)},
            #{"role":"user", "content":text"}
        ]
    )
    finalResponse = response.choices[0].message.content
    return finalResponse

#papago
def transPapago(text, PAPAGO_ID, PAPAGO_PW):
    data = {'text': text,
            'source': 'en',
            'target': 'ko'}
    
    url = 'https://openapi.naver.com/v1/papago/n2mt'
    headers = {'X-Naver-Client-Id': PAPAGO_ID, 
               'X-Naver-Client-Secret': PAPAGO_PW}
    
    response = requests.post(url, data=data, headers=headers)
    rescode = response.status_code
    
    if(rescode == 200):
        send_data = response.json() 
        trans_data = (send_data['message']['result']['translatedText']) 
        return trans_data 
    else:
        print("Error Code:" + rescode)

#deepl
def transDeepl(text, apikey):
    trans = deepl.Translator(apikey)
    result = trans.translate_text(text, target_lang='ko')
    return(result) 

#google
def transGoogle(message):
    google = Translator()
    result = google.translate(message, dest='ko')
    return result.text

#main
def main():
    st.set_page_config(
        page_title="Translator App",
        page_icon="🌎",
        layout="wide",
    )

    #session_state 초기화
    if "OPENAI_API" not in st.session_state:
        st.session_state["OPENAI_API"] = ""
    if "PAPAGO_ID" not in st.session_state:
        st.session_state["PAPAGO_ID"] = ""
    if "PAPAGO_PW" not in st.session_state:
        st.session_state["PAPAGO_PW"] = ""
    if "DEEPL_API" not in st.session_state:
        st.session_state["DEEPL_API"] = ""

    #sidebar
    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input(label = "OPENAI_API", placeholder="Enter your OPENAI_API", type="password")
        st.markdown('---')

        st.session_state["PAPAGO_ID"] = st.text_input(label = "PAPAGO_ID", placeholder="Enter your PAPAGO_ID", type="password")
        st.session_state["PAPAGO_PW"] = st.text_input(label = "PAPAGO_PW", placeholder="Enter your PAPAGO_PW", type="password")
        st.markdown('---')

        st.session_state["DEEPL_API"] = st.text_input(label = "DEEPL_API", placeholder="Enter your DEEPL_API", type="password")

    st.header(':earth_asia:번역 플랫폼 비교하는 프로그램:earth_asia:')
    st.markdown('---')

    st.subheader('번역할 텍스트 입력하시오')
    text = st.text_area(label='', placeholder='Input your text in English', height=200) #text.input은 한줄만 입력받는다.
    st.button('번역하기')
    st.markdown('---')

    col1, col2 = st.columns(2)

    with col1 :
        st.subheader('ChatGPT 번역')
        if st.session_state["OPENAI_API"] and text: #and : 논리연산자
            result = tranGPT(text, st.session_state["OPENAI_API"])
            st.info(result) #텍스트 뷰어
        else :
            st.info('번역할 텍스트와 API를 입력하세요.')
    
    with col2:
        st.subheader('Papago 번역')
        if st.session_state["PAPAGO_ID"] and st.session_state["PAPAGO_PW"] and text:
            result = transPapago(text, st.session_state["PAPAGO_ID"], st.session_state["PAPAGO_PW"])
            st.info(result)
        else :
            st.info('번역할 텍스트와 ID, PW를 입력하세요.')

    st.markdown('---')

    col3, col4 = st.columns(2)

    with col3 :
        st.subheader('Deepl 번역')
        if st.session_state["DEEPL_API"] and text:
            result = transDeepl(text, st.session_state["DEEPL_API"])
            st.info(result)
        else :
            st.info('번역할 텍스트와 API를 입력하세요.')
            
    with col4:
        st.subheader('Google 번역')
        if text:
            result = transGoogle(text)
            st.info(result)
        else :
            st.info('번역할 텍스트를 입력하세요.')
    

if __name__ == '__main__':
    main()