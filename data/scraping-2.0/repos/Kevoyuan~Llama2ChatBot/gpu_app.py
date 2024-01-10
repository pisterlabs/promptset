models = {'Llama-2-7b Chat (q4_K_S)': 'llama-2-7b-chat.ggmlv3.q4_K_S', 'Llama-2-7b Chat (q2_K)': 'llama-2-7b-chat.ggmlv3.q2_K', 'Llama-2-13b Chat (q4_0)': 'llama-2-13b-chat.ggmlv3.q4_0', 'Llama-2-13b Chat (q8_0)': 'llama-2-13b-chat.ggmlv3.q8_0'}

# app.py
from typing import List, Union
import os
from streamlit_extras.app_logo import add_logo
from streamlit_extras.badges import badge

# from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
def run_mlc_chat_cli(model_id: str) -> None:
    command = ["mlc_chat_cli", "--local-id", model_id]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode("utf-8"), stderr.decode("utf-8")

import subprocess
from download_models import download_model_if_not_exist




def init_page() -> None:
    
    st.set_page_config(page_title="Llama 2", page_icon="logo/llama-144.png")
    add_logo("logo/llama-144.png",height=300)
    

    
    st.header("🦙 Llama 2 ChatBot")
    hide_menu_style = '''
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        '''
    st.markdown(hide_menu_style, unsafe_allow_html=True)
def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear_conversation")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Respond your answer in mardkown format."
            )
        ]
        # st.session_state.costs = []


def select_llm() -> Union[ChatOpenAI, LlamaCpp]:

    # Model selection in the sidebar
    with st.sidebar:
        model_name = st.radio(
            "Choose LLM:",
            (
                "llama-2-7b-chat.ggmlv3.q4_K_S",
                "llama-2-7b-chat.ggmlv3.q2_K",
                "llama-2-13b-chat.ggmlv3.q4_0",
                "llama-2-13b-chat.ggmlv3.q8_0",
                
                #  "open-llama-7B-open-instruct.ggmlv3.q4_0.bin"
            ),
        )

    # Dictionary to map model names to their respective URLs and file names
    MODEL_URLS = {
        "llama-2-7b-chat.ggmlv3.q2_K": "https://huggingface.co/localmodels/Llama-2-7B-Chat-ggml/resolve/main/llama-2-7b-chat.ggmlv3.q2_K.bin",
        "llama-2-13b-chat.ggmlv3.q8_0": "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q8_0.bin",
        # "open-llama-7B-open-instruct.ggmlv3.q4_0.bin": "https://huggingface.co/TheBloke/open-llama-7b-open-instruct-GGML/resolve/main/open-llama-7B-open-instruct.ggmlv3.q4_0.bin",
        "llama-2-13b-chat.ggmlv3.q4_0": "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin",
        "llama-2-7b-chat.ggmlv3.q4_K_S": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_K_S.bin",
    }

    # Get the URL and file name based on the selected model name
    url = MODEL_URLS[model_name]
    file_name = os.path.basename(url)
    models_dir = "./models"

    # Call the function to download the model if it doesn't exist
    downloaded = download_model_if_not_exist(url, file_name, models_dir)

    # if not downloaded:
    #     st.sidebar.success("Model already exists. No need to download.")

    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.01, max_value=5.0, value=0.1, step=0.01
    )
    top_p = st.sidebar.slider(
        "Top P:", min_value=0.01, max_value=1.0, value=0.9, step=0.01
    )
    max_seq_len = st.sidebar.slider(
        "Max Sequence Length:", min_value=64, max_value=4096, value=2048, step=8
    )
    n_ctx = st.sidebar.slider(
        "Adjusting the Context Window:",
        min_value=512,
        max_value=20480,
        value=512,
        step=512,
    )

   
    if model_name.startswith("llama-2-"):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path=f"./models/{model_name}.bin",
            n_ctx=n_ctx,
            input={
                "temperature": temperature,
                "max_length": 2000,
                #    "n_ctx": 2048,
                "top_p": top_p,
                "repetition_penalty": 1,
                "max_seq_len": max_seq_len,
            },
            callback_manager=callback_manager,
            verbose=False,  # True
        )
    if st.sidebar.button("Run MLC Chat CLI"):
        stdout, stderr = run_mlc_chat_cli(model_name)
        st.sidebar.text(stdout)
        if stderr:
            st.sidebar.text("Error:")
            st.sidebar.text(stderr)
        raise ValueError(f"Unsupported model: {model_name}")


def get_answer(llm, messages) -> tuple[str, float]:

    if isinstance(llm, LlamaCpp):
        return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages)))

    else:
        raise TypeError(f"Unsupported llm type: {type(llm).__name__}")
    # Using run_mlc_chat_cli with the selected model
    stdout, stderr = run_mlc_chat_cli(llm)
    if stderr:
        return f"Error: {stderr}"



def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchainschema_to_dict(
    messages: List[Union[SystemMessage, HumanMessage, AIMessage]]
) -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [
        {"role": find_role(message), "content": message.content} for message in messages
    ]


def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


def main() -> None:
 

    init_page()
    llm = select_llm()
    init_messages()

    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking ..."):
      
            answer = get_answer(llm, st.session_state.messages)

        st.session_state.messages.append(AIMessage(content=answer))


    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)




# streamlit run app.py
if __name__ == "__main__":
    main()
    with st.sidebar:
        badge(type="github", name="Kevoyuan/chatbot")
        
