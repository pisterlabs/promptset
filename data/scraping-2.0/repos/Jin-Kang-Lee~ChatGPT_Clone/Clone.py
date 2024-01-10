from openai import OpenAI
import streamlit as st


st.title("ChatGPT-Mimic")

#Set the OpenAI API Key from Streamlit - file is being pulled from secrets.toml
    #!!!GENERATE AND USE YOUR OWN API KEY!!!
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

#Set a GPT default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"


#Init the chat history
    #Check if messages exist in session_state, if not we initialize it to empty list
if "messages" not in st.session_state:
    st.session_state.messages = []


#Go through and start writing our already existing messages
    #This is a way to store chat history so we can display it in the chat message container
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


#Take in user input
    #prompt variable will check if user sent message, will display message in container and append to chat history
if prompt := st.chat_input("Ask me anything..."):

    #Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    #Writes out your message/prompt on the screen
    with st.chat_message("user"):
        st.markdown(prompt)
    
    #Writes out chatGPT's reply to your message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        #Initial full response is empty
        full_response = ""

        for response in client.chat.completions.create( 
            #TESTTT
            #Plug in the chatgpt model we are using
                #Set stream to true as it will reply to our prompt live
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            #Append full_response whenever we get a message from openai
            full_response += (response.choices[0].delta.content or "")

            #Adding effect to make it look like the assistant is thinking
            message_placeholder.markdown(full_response + "▌")

        #Once done we render out the full_response
        message_placeholder.markdown(full_response)
    #Append and call full_response
    st.session_state.messages.append({"role": "assistant", "content": full_response})