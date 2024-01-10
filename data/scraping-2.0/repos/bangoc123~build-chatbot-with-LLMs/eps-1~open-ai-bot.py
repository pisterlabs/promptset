import streamlit as st
from openai import OpenAI
st.title("ProtonX x ChatGPT")
link='All slides in here [link](https://protonx.io/courses/63b4e2120c745e00190cb39a)'
st.markdown(link,unsafe_allow_html=True)
import time

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in  st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Hãy nhập vào yêu cầu?"):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        full_res = ""
        holder = st.empty()

        # for word in prompt.split():
        #     full_res += word + " "
        #     time.sleep(0.05)
        #     holder.markdown(full_res + "▌")
        # holder.markdown(full_res)

        for response in client.chat.completions.create(
            model = st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_res += (response.choices[0].delta.content or "")
            holder.markdown(full_res + "▌")
            holder.markdown(full_res)
        holder.markdown(full_res)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_res
        }
    )