# comment
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage
import streamlit as st
def main():
    inference_modifier = {
        "temperature": 1,
        "top_p": .999,
        "top_k": 250,
        "max_tokens_to_sample": 300,
        "stop_sequences": ["\n\nHuman:"]
    }
    chat = BedrockChat(model_id="anthropic.claude-v2", model_kwargs=inference_modifier)
    messages = [
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        )
    ]
    chat(messages)

    user_input = st.text_area("Enter querry to Bedrock")
    button = st.button("Ask Bedrock")
    messages = [
        HumanMessage(
            content=f"{user_input}"
        )
    ]
    if user_input and button:
        summary = chat(messages)
        st.write("Summary : ", summary)

if __name__ == "__main__":
    main()
    
# streamlit run basic_streamlit_app_bedrock.py 