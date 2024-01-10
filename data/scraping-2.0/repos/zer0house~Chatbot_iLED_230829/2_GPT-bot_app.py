# openai module 설치: pip install openai
import openai
import streamlit as st

st.title("GPT-Bot")

# .streamlit 폴더에 secrets.toml 파일을 만들고 아래와 같이 OPENAI_API_KEY를 저장합니다.
# .gitignore 파일에 .streamlit/secrets.toml을 추가합니다.
openai.api_key = st.secrets["OPENAI_API_KEY"]

# openai_model = "gpt-3.5-turbo" 또는 "gpt-4" # 소문자로만 써야합니다!
if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"


with st.chat_message(name="assistant", avatar="https://avatars.githubusercontent.com/u/78703832?v=4"):
    st.write("Hello 😀")
    st.write("I'm GPT Bot. I can answer your questions about GPT-3.5 Turbo.")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        

prompt = st.chat_input("What is up?")
if prompt: 
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send user message to OpenAI API
    api_response = openai.ChatCompletion.create(
        model=st.session_state.openai_model,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages # 0번째부터 m번째까지 반복
        ],
        max_tokens=256, # max_tokens는 1과 2048 사이의 정수여야 합니다. # max_tokens는 생성할 토큰의 최대 개수를 조절하는 파라미터입니다.
        temperature=1, # temperature는 0과 1 사이의 값이어야 합니다. # temperature는 토큰의 확률분포의 엔트로피를 조절하는 파라미터입니다.
        top_p=0.5, # top_p는 0과 1 사이의 값이어야 합니다. # top_p는 토큰의 확률분포의 상위 p%만을 고려하는 파라미터입니다.
        # frequency_penalty=0,
        # presence_penalty=0,
    )
    # Extract assistant response from API response
    response = api_response.choices[0].message.content # 왜 이런 구조인지 확인해봅시다.
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

