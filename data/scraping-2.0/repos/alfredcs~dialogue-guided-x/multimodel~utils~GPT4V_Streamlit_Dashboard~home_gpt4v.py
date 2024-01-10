import streamlit as st
from PIL import Image
#import google.generativeai as genai
from openai import OpenAI

st.set_page_config(page_title="GPT-4V with Streamlit",page_icon="🩻")

st.write("Welcome to the GT-4V Dashboard. You can proceed by providing your OpenAI API Key")

with st.expander("Provide Your OpenAI API Key"):
     openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
     
if not openai_api_key:
    st.info("Enter the OpenAI API Key to continue")
    st.stop()

genai.configure(api_key=openai_api_key)
client = OpenAI()

st.title("GPT-4V with Streamlit Dashboard")

with st.sidebar:
    option = st.selectbox('Choose Your Model',('gpt-4-turbo', 'gpt-4-vision-preview'))

    if 'model' not in st.session_state or st.session_state.model != option:
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.model = option
    
    st.write("Adjust Your Parameter Here:")
    temperature = st.number_input("Temperature", min_value=0.0, max_value= 1.0, value =0.5, step =0.01)
    max_token = st.number_input("Maximum Output Token", min_value=0, value =100)
    gen_config = genai.types.GenerationConfig(max_output_tokens=max_token,temperature=temperature)

    #st.divider()
    #st.markdown("""<span ><font size=1>Connect With Me</font></span>""",unsafe_allow_html=True)
    #"[Linkedin](https://www.linkedin.com/in/cornellius-yudha-wijaya/)"
    #"[GitHub](https://github.com/cornelliusyudhawijaya)"
    
    st.divider()
    
    upload_image = st.file_uploader("Upload Your Image Here", accept_multiple_files=False, type = ['jpg', 'png'])
    
    if upload_image:
        image = Image.open(upload_image)
    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

 
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if upload_image:
    if option == "gpt-4-turbo":
        st.info("Please Switch to the gpt-4-vision-preview")
        st.stop()
    if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            response=st.session_state.chat.send_message([prompt,image],stream=True,generation_config = gen_config)
            response.resolve()
            msg=response.text

            st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
            st.session_state.messages.append({"role": "assistant", "content": msg})
            
            st.image(image,width=300)
            st.chat_message("assistant").write(msg)

else:
    if prompt := st.chat_input():
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            response=st.session_state.chat.send_message(prompt,stream=True,generation_config = gen_config)
            response.resolve()
            msg=response.text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
    
    
