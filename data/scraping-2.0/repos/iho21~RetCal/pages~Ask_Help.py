import streamlit as st
from openai import OpenAI
import time
import openai

st.title('Financial Advisor')

# st.write(st.session_state)
openai_api_key = st.text_input("OpenAI Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "I am your Financial Advisor. I can help you with your financial planning."},
        {"role": "assistant", "content": "How can I assist you today?"}
    ]

for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.chat_message('User').write(f'User: {message["content"]}')
    else:
        st.chat_message('assistant').write(f'Assistant: {message["content"]}')

i = 1
user_message = st.text_input("Your message", key=f"message{i}")
i = 2
while True:
    if user_message:
        st.session_state["messages"].append({"role": "user", "content": user_message})

        if openai_api_key:
            try:
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                  model="gpt-3.5-turbo",
                  messages=st.session_state["messages"]
                )
                assistant_message = response.choices[0].message
                print(assistant_message)
                st.session_state["messages"].append({"role" : "assistant", "content":assistant_message.content}) #{"role": "assistant", "content": assistant_message})
                st.chat_message("assistant").write(assistant_message.content)
            except Exception as e:
                st.chat_message("assistant").write("Sorry, an error occurred. Try again in a moment.")
                st.write(e)
                st.stop()
            except openai.RateLimitError:
                st.chat_message("assistant").write("LImit crossed")
                st.stop()
            time.sleep(5)
            user_message = st.text_input("Your message", key=f"message{i}")
            st.session_state["messages"].append({"role": "user", "content": user_message})
            i+=2
        else:
            st.chat_message("assistant").write("Please add your OpenAI API key.")
