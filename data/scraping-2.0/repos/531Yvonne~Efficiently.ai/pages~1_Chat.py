import streamlit as st
import openai
import csv
import json
from pages.tools.moderation import moderate_text

st.title("💬 Chatbot")
st.caption("🚀 A customized efficiency assistant powered by OpenAI")

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

with st.sidebar:
    if not st.session_state["openai_api_key"]:
        st.info("Please add your OpenAI API key to continue.")
        st.session_state["openai_api_key"] = st.text_input("OpenAI API Key",
                                                           "")
    else:
        st.text_input("Current OpenAI API Key",
                      value=st.session_state["openai_api_key"],
                      type="password")

    openai.api_key = st.session_state["openai_api_key"]

    # Response Language Selection
    response_language = st.sidebar.selectbox(
        "Select Response Language",
        ["English", "French", "Spanish", "Chinese", "Japanese", "Korean"])

    # Export History Record
    # Format Selection
    export_format = st.selectbox(
        "Select Export Format", ["JSON", "CSV", "TXT"])

    # Export Chat History Button
    if st.sidebar.button("Export Chat History"):
        if export_format == "JSON":
            with open("chat_history.json", mode="w") as file:
                json.dump(st.session_state.messages, file, indent=4)
        elif export_format == "CSV":
            with open("chat_history.csv", mode="w", newline="") as file:
                writer = csv.writer(file)
                for msg in st.session_state.messages:
                    writer.writerow([msg["role"], msg["content"]])
        elif export_format == "TXT":
            with open("chat_history.txt", mode="w") as file:
                for msg in st.session_state.messages:
                    file.write(f"{msg['role']}: {msg['content']}\n")
        st.success(
            f"Chat history exported as {export_format} to the app's root!")

    # Clear Chat History Feature
    if st.button("Clear Chat History"):
        st.session_state["messages"] = [
            {"role": "assistant",
             "content": "Messages Cleared. How can I help you again?"},
            {"role": "system",
             "content": '''You are a professional assistant designed to provide
             accurate information and support. Process questions step by step,
             double check whether conditions are satisfied, and provide
             response in a concise, well-structured, ordered format.
             '''}]
        st.success("Chat history cleared!")

# Initial Chat Welcome Text
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant",
         "content": "How can I help you?"},
        {"role": "system",
         "content": '''You are a professional assistant designed to provide
             accurate information and support. Process questions step by step,
             double check whether conditions are satisfied, and provide
             response in a concise, well-structured, ordered format.
             '''}]
# Show all messages in a chat container
for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not st.session_state["openai_api_key"]:
        st.info("Please add your OpenAI API key at the sider bar to continue.")
        st.stop()
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(
            {"role": "user",
             "content": f"{prompt}, response_language={response_language}"})
    # Add Moderation Layer for Responsible AI purpose
    screening_result = moderate_text(prompt)
    if screening_result["flag"]:
        msg = f'''Unable to finish your request.
        Your request contains: {screening_result['content']} content'''
        st.chat_message("assistant").write(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        with st.spinner("Thinking..."):
            response = openai.chat.completions.create(model="gpt-3.5-turbo",
                                                      messages=st.session_state.messages)
        msg = response.choices[0].message.content
        st.chat_message("assistant").write(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
