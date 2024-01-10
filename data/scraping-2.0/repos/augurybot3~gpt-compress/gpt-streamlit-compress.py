import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()
# client.api_key = os.getenv('OPENAI_API_KEY')
default_model = "gpt-3.5-turbo-1106"
client.api_key = st.secrets["OPENAI_API_KEY"]


def encode_and_chat(data_sample, encode_prompt, system_prompt):
    encoded_user_prompt = f"""{encode_prompt}\n```data\n{data_sample}\n```"""
    encode_messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': encoded_user_prompt}
    ]
    return chat(encode_messages)

def decode_and_chat(encoded_data, incept_msg, system_prompt):
    decoded_user_prompt = f"""
    - From the compressed data, fully unpack, decompress and transform the information back into its original state...
    ```
    {encoded_data}
    ```
    """
    decode_messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': incept_msg},
        {'role': 'assistant', 'content': encoded_data},
        {'role': 'user', 'content': decoded_user_prompt}
    ]
    return chat(decode_messages)

def chat(messages, model=default_model):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

def main():
    st.title("AI Compression and Data Storage Expert")

    system_prompt = "You are an efficient AI Compression And Data Storage Expert..."
    encode_prompt = "encode a compressed, shortened version of the given data..."
    incept_msg = "please display the encoded and compressed data from your previous inference cycle."

    data = st.file_uploader("Upload data for encoding", type=['txt', 'json', 'csv'])

    if data is not None:
        data_sample = data.getvalue().decode("utf-8")
        
        

        if st.button("Encode"):
            encoded_data = encode_and_chat(data_sample, encode_prompt, system_prompt)
            st.text_area("Encoded Data", encoded_data, height=450)
            if 'encoded_data' not in st.session_state:
                st.session_state.encoded_data = encoded_data
            decoded_data = decode_and_chat(st.session_state.encoded_data, incept_msg, system_prompt)
            st.text_area("Decoded Data", decoded_data, height=450)

if __name__ == "__main__":
    main()
