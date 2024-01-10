import openai
import streamlit as st
# from flask_cors import CORS

# Enable CORS for all routes
# CORS(st)

def response_msg(openai_api_key: str, prompt: str):
    st.session_state.messages.append({"role": user_icon, "content": prompt})
    st.chat_message(user_icon).write(prompt)  # Change user icon
    result = {
        "role": "assistant",
        "content": "Thank you for your message and support! We are not yet public.",
    }
    if openai_api_key:
        openai.api_key = openai_api_key
        openai_msg = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=st.session_state.messages
        ).choices[0].message
        result = {"role": "assistant", "content": openai_msg.content}  # Change bot icon
    return result

# Custom CSS for centering the title
st.markdown(
    """
    <style>
    .css-1lqj2e3 {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Assuming your custom icon is saved as "custom_icon.png"
custom_icon_path = "assets/aquachat_assistant_icon01.2.jpg"

def assistant_message(content):
    col1, col2 = st.columns([0.075, 1])  # Adjust the width ratio between icon and text as needed
    col1.image(custom_icon_path, width=25)
    col2.write(content)

# Inside your code where you display assistant's messages
# st.image(custom_icon_path, use_column_width=True)

with st.sidebar:
    user_icon = st.text_input("Custom User Icon", "👤")  # Set default value here
    openai_api_key = None

st.title("💬 Aquadviser")
st.caption("An Aqua Chatbot powered by Aquadviser")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        assistant_message(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    msg = response_msg(openai_api_key, prompt)
    st.session_state.messages.append(msg)
    assistant_message(msg["content"])
