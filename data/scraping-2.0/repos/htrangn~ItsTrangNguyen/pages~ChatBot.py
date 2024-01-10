import openai
import streamlit as st



with st.sidebar:
    activities2 = ["Chat", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities2)
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[OpenAI's Platform website](https://platform.openai.com/account/api-keys)"
    "[Instruct to get an OpenAI API key](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/)"
    st.write("View the source code:   ", "[GITHUB](https://github.com/htrangn/ItsTrangNguyen/edit/main/pages/ChatBot.py)") 
    markdown = """
    Web App URL: <https://facapp.streamlit.app>
    """
    st.sidebar.info(markdown)

if choice == "Chat":
    st.title("💬 CHATBOT")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
    
        openai.api_key = openai_api_key
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        msg = response.choices[0].message
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg.content)
elif choice == "About":
     st.subheader("About this app")
     st.write("This app was made by Nguyen H.Trang")
     st.write("This app requires an OpenAI API key to activate")
     st.write("This chatbot is designed to deliver a seamless conversational experience with its natural language processing capabilities")
