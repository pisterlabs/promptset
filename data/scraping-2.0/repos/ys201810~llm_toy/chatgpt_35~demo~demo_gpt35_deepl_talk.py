# coding=utf-8
import openai
import pathlib
import streamlit as st
from chatgpt_35.utils import get_config
from chatgpt_35.utils.openai_wrapper import ChatGPT
from chatgpt_35.utils.deepl_wrapper import DeepL

base_path = pathlib.Path.cwd().parent.parent
config_file = base_path / 'config' / 'config.yaml'
config = get_config.run(config_file)

prompt_template = """
- Make the following statement more attractive for company reviewer.

Human: {input}
Assistant:
"""

def main():
    st.title("ChatGPT Demo using Streamlit")

    # Initialize conversation_history in session_state if not exists
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    input_text = st.text_input("Enter your message:")
    deepl = DeepL(config.deepl_api_key, config.deepl_url)
    chatgpt = ChatGPT(config.openai_api_key, config.openai_model)

    if st.button("Send"):
        # Add user's message to conversation history
        st.session_state.conversation_history.append({"sender": "User", "message": input_text})

        normal_prompt = f"""
        以下の文章を魅力的にしてください。

        {input_text}
        """.format(input_text)
        normal_response = chatgpt.run_chat(normal_prompt)

        # Add AI's response to conversation history
        st.session_state.conversation_history.append({"sender": "AI(No DEEPL)", "message": normal_response})

        # 入力された日本語を英語に翻訳
        translate_result = deepl.run_translate(
            text=input_text,
            from_lang='JA',
            to_lang='EN'
        )

        # 翻訳された英語のテキストをchatGPTに投入
        prompt = prompt_template.format(**{'input': translate_result})
        gpt_response = chatgpt.run_chat(prompt)

        # chatGPTのレスポンスを日本語に翻訳
        translate_result = deepl.run_translate(
            text=gpt_response,
            from_lang='EN',
            to_lang='JA'
        )

        # Add AI's response to conversation history
        st.session_state.conversation_history.append({"sender": "AI(with DEEPL)", "message": translate_result})

        # Keep only the last 10 messages
        st.session_state.conversation_history = st.session_state.conversation_history[-12:]

    # Display conversation history
    st.write("Conversation history:")
    for msg in st.session_state.conversation_history:
        st.write(f"{msg['sender']}: {msg['message']}")

if __name__ == '__main__':
    main()