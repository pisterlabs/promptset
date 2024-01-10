import streamlit as st
import openai
import config as cfg
import json
import os

st.set_page_config(layout="wide")

st.markdown(
    """<style>.block-container{max-width: 86rem !important;}</style>""",
    unsafe_allow_html=True,
)


def save_chats_to_file(chats_st):
    # Exclude streaming and generator states from the saved chats
    chats_to_save = {
        chat_name: {
            key: value
            for key, value in chat.items()
            if key not in ("streaming", "generator")
        }
        for chat_name, chat in chats_st.items()
    }
    with open(cfg.CHATS_SAVE_FILE, "w") as f:
        json.dump(chats_to_save, f)


def load_chats_from_file():
    if os.path.exists(cfg.CHATS_SAVE_FILE):
        with open(cfg.CHATS_SAVE_FILE, "r") as f:
            chats_st = json.load(f)
        for chat_name in chats_st:
            chats_st[chat_name]["streaming"] = False
            chats_st[chat_name]["generator"] = None

        return chats_st

    return {}


def init_session():
    # Load saved chats from file
    saved_chats = load_chats_from_file()

    # Session state initialization
    if "chats" not in st.session_state:
        st.session_state.chats = saved_chats
    if "params" not in st.session_state:
        st.session_state.params = dict()
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = None


def new_chat(chat_name):
    if chat_name not in st.session_state.chats:
        st.session_state.chats[chat_name] = {
            "chat_history": [],
            "streaming": False,
            "generator": None,
        }
    else:
        chat_name = f"{chat_name} (copy)"
        st.session_state.chats[chat_name] = {
            "chat_history": [],
            "streaming": False,
            "generator": None,
        }
    st.session_state.current_chat = chat_name
    # Save chats to file
    save_chats_to_file(st.session_state.chats)


def chat(
    messages,
    n=1,
    stream=True,
):
    """
    Generate chat responses using the OpenAI API.

    Args:
        messages (list): A list of messages to be processed.
        max_tokens (int): The maximum length of the output message.
        temperature (float, optional): Controls the randomness of the model's
        output. Defaults to 1.
        n (int, optional): The number of completions to generate for each prompt.
        Defaults to 1.
        model (str, optional): The model to use for generating responses. Defaults
        to "gpt-3.5-turbo-16k".
        stream (bool, optional): Whether to stream the output. Defaults to False.

    Yields:
        str: The generated response.
    """

    completion = openai.chat.completions.create(
        model=cfg.OPENAI_MODELS[st.session_state["params"]["model"]]["model_name"],
        messages=messages,
        max_tokens=st.session_state["params"]["max_tokens"],
        temperature=st.session_state["params"]["temperature"],
        n=n,
        stream=stream,
    )
    for chunk in completion:
        try:
            yield chunk.choices[0].delta.content if chunk.choices[
                0
            ].finish_reason != "stop" else ""
        except:
            yield "error!"


def render_sidebar(models_list):
    with st.sidebar:
        # New chat button
        if st.button("New Chat"):
            chat_name = f"Chat{len(st.session_state.chats) + 1}"
            new_chat(chat_name)
            st.session_state.selected_chat = chat_name
            st.session_state.editable_chat_name = chat_name
            st.session_state.current_chat = chat_name

        # Chat selection
        chat_names = list(st.session_state.chats.keys())
        if not chat_names:
            st.session_state.selected_chat = None
            st.session_state.current_chat = None
            st.session_state.editable_chat_name = ""
        else:
            if "selected_chat" not in st.session_state:
                st.session_state.selected_chat = chat_names[0] if chat_names else None

            selected_chat_index = (
                chat_names.index(st.session_state.selected_chat)
                if st.session_state.selected_chat
                else 0
            )
            selected_chat = st.selectbox(
                "Select a chat",
                chat_names,
                index=selected_chat_index,
                key="chat_selection",
                on_change=update_selected_chat,
                args=(chat_names,),
            )

    with st.sidebar:
        st.markdown("---")
        # GPT Model Configuration
        st.session_state.params["model"] = st.selectbox(
            "Select a model",
            models_list,
            index=0,
        )
        st.session_state.params["temperature"] = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )
        max_tokens = cfg.OPENAI_MODELS[st.session_state.params["model"]]["max_tokens"]
        st.session_state.params["max_tokens"] = st.slider(
            "Max Tokens", min_value=1, max_value=max_tokens, value=1000, step=1
        )

    with st.sidebar:
        st.markdown("---")
        # Chat name editing
        if st.session_state.current_chat:
            new_chat_name = st.text_input(
                "Edit chat name",
                st.session_state.editable_chat_name,
                key="chat_name_input",
            )

            if st.button("Rename Chat"):
                if new_chat_name and new_chat_name != st.session_state.current_chat:
                    st.session_state.chats[new_chat_name] = st.session_state.chats.pop(
                        st.session_state.current_chat
                    )
                    st.session_state.selected_chat = new_chat_name
                    st.session_state.editable_chat_name = new_chat_name
                    st.session_state.current_chat = new_chat_name
                    save_chats_to_file(st.session_state.chats)

    with st.sidebar:
        st.markdown("---")

        # embeddings_db_list = list(cfg.EMBEDDINGS_DBS.keys())
        embeddings_db_list = ["none", "bbg_chat", "cvm"]
        st.session_state.params["embeddings"] = st.selectbox(
            "Select an context database",
            embeddings_db_list,
            index=0,
        )

    with st.sidebar:
        st.markdown("---")

        # Delete current chat
        if st.session_state.current_chat and st.button("Delete Current Chat"):
            delete_chat(st.session_state.current_chat)


def delete_chat(chat_name):
    if chat_name in st.session_state.chats:
        del st.session_state.chats[chat_name]
        # if chat_name == st.session_state.selected_chat:
        #     st.session_state.selected_chat = None
        # if chat_name == st.session_state.current_chat:
        #     st.session_state.current_chat = None
        st.session_state.selected_chat = None
        st.session_state.current_chat = None
        st.session_state.editable_chat_name = ""
        save_chats_to_file(st.session_state.chats)
        st.rerun()


def run_chat_interface():
    current_chat = st.session_state.current_chat
    if current_chat is None:
        st.warning("Please create a new chat or select an existing one.")
        return

    chat_data = st.session_state.chats[current_chat]
    chat_history = chat_data["chat_history"]
    create_chat_area(chat_history)

    # Chat controls
    user_input = st.chat_input("Ask something")

    # Handle user input and generate assistant response
    if user_input or chat_data["streaming"]:
        process_user_input(user_input, chat_data)


def create_chat_area(chat_history):
    # Display the chat history
    for c in chat_history:
        role = c["role"]
        with st.chat_message(role):
            st.write(c["content"])


def process_user_input(user_input, chat_data):
    # Process the user input and generate assistant response
    chat_history = chat_data["chat_history"]
    if user_input:
        chat_history.append({"role": "user", "content": user_input})
        gpt_answer = chat(chat_history)
        chat_data["generator"] = gpt_answer
        chat_data["streaming"] = True
        chat_history.append({"role": "assistant", "content": ""})
        # Save chats to file
        save_chats_to_file(st.session_state.chats)
    else:
        update_assistant_response(chat_data)

    st.rerun()


def update_selected_chat(chat_names):
    new_selected_chat = st.session_state.chat_selection
    if new_selected_chat in chat_names:
        st.session_state.selected_chat = new_selected_chat
        st.session_state.current_chat = new_selected_chat
        st.session_state.editable_chat_name = new_selected_chat


def update_assistant_response(chat_data):
    try:
        chunk = next(chat_data["generator"])
        chat_data["chat_history"][-1]["content"] += chunk
        # Save chats to file
        save_chats_to_file(st.session_state.chats)
        st.rerun()
    except StopIteration:
        chat_data["streaming"] = False
        # Save chats to file
        save_chats_to_file(st.session_state.chats)
        st.rerun()


def main():
    models_list = list(cfg.OPENAI_MODELS.keys())
    init_session()
    render_sidebar(models_list)
    run_chat_interface()


if __name__ == "__main__":
    main()
