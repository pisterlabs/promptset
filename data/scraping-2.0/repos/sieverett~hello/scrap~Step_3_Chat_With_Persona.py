from openai import OpenAI
import streamlit as st
from utils import (capitalize_names, list_cloned_voices)
import pandas as pd 

# Initialize OpenAI client
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=st.secrets["OPENAI_API_KEY"],
)

st.write(st.session_state)


# @st.cache(allow_output_mutation=True)

# st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")


cloned_dict=list_cloned_voices()
cloned_names=list(cloned_dict.keys())
cloned_names_caps=capitalize_names(cloned_names)
cloned_names_caps=[n for n in cloned_names_caps if n not in ('Ray Dalio','Dalai Lama')]
selected_name = st.sidebar.selectbox("Select a TED Presenter", cloned_names_caps)
selected_name_lower = selected_name.lower()
voice_id = cloned_dict[selected_name_lower]

st.title(f"💬 Chat With {selected_name}")

df = pd.read_csv("TED_playlist_info.csv")
transcript = df[df.presenter==selected_name_lower]['transcripts'].to_string(index=False)[400:]

st.write(transcript)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What's on your mind?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

st.write(st.session_state.messages)

if prompt := st.chat_input():
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    st.write(st.session_state.messages)
    # response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    # prompt = f"Following a presentation you gave, an audience member asks: '{prompt}'\n\nPresentation: {transcript}\n\nResponse: "
    response = client.chat.completions.create(
        messages=st.session_state.messages,
        model="gpt-3.5-turbo",
    )
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)