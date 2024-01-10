import openai
import pandas as pd
import streamlit as st
import os
from streamlit_chat import message
from sklearn.metrics.pairwise import cosine_similarity
from utils.ask import ask
from utils.num_tokens import num_tokens


def render_chat_ui(extracted_text):
    token_count = num_tokens(extracted_text)
    # styl = f"""
    # <style>
    #     .stTextInput {{
    #     position: fixed;
    #     bottom: 3rem;
    #     }}
    # </style>
    # """
    # st.markdown(styl, unsafe_allow_html=True)

    # Set org ID and API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model='gpt-4'

    EMBEDDING_MODEL = (
        "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
    )
    #init df
    df = pd.DataFrame({"text": [extracted_text], "embedding": [extracted_text]})
    if(token_count > 4096):
        try:
            chunk_size = 4096  # Define the desired chunk size
            chunks = []
            start_idx = 0
            while start_idx < len(extracted_text):
                end_idx = min(start_idx + chunk_size, len(extracted_text))
                chunk = extracted_text[start_idx:end_idx]
                chunks.append(chunk)
                start_idx = end_idx
            embeddings = []

            for chunk in chunks:
                response = openai.Embedding.create(model=EMBEDDING_MODEL, input=chunk)
                for i, be in enumerate(response["data"]):
                    assert (
                        i == be["index"]
                    )  # double check embeddings are in same order as input
                batch_embeddings = [e["embedding"] for e in response["data"]]
                embeddings.extend(batch_embeddings)

            df = pd.DataFrame({"text": chunks, "embedding": embeddings})

            # st.dataframe(df)
            SAVE_PATH = "data/finc-taxes.csv"
            df.to_csv(SAVE_PATH, index=False)
        except Exception as e:
            st.write(e)

    def generate_response(prompt):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        # print(st.session_state["messages"])

        completion = openai.ChatCompletion.create( model=model, messages=st.session_state["messages"] )
        response = completion.choices[0].message.content
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # print(st.session_state['messages'])
        total_tokens = completion.usage.total_tokens
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        return response, total_tokens, prompt_tokens, completion_tokens
    
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input("You:", key="input")
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            if token_count + len(user_input.split()) < 4096:
                output, total_tokens, prompt_tokens, completion_tokens = generate_response(
                    user_input
                )
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(output)
                st.session_state["total_tokens"].append(total_tokens)

                # from https://openai.com/pricing#language-models
                cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
            elif token_count > 4096:
                output = ask(user_input, df)
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(output)

    # Initialise session state variables
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "system",
                "content": "You answer questions about this document:"
                + "\n"
                + extracted_text,
            }
        ]
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = []
    if "cost" not in st.session_state:
        st.session_state["cost"] = []
    if "total_tokens" not in st.session_state:
        st.session_state["total_tokens"] = []
    if "total_cost" not in st.session_state:
        st.session_state["total_cost"] = 0.0

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    # st.sidebar.title("Sidebar")
    # model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    # counter_placeholder = st.sidebar.empty()
    # counter_placeholder.write(
    #     f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}"
    # )
    clear_button = st.button("Clear Conversation", key="clear")

    # Map model names to OpenAI model IDs
    # if model_name == "GPT-3.5":
        # model = "gpt-4"
    # else:
    model = "gpt-4"

    # reset everything
    if clear_button:
        st.session_state["generated"] = []
        st.session_state["past"] = []
        if(token_count > 4096):
            st.session_state["messages"] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
        else:
            st.session_state["messages"] = [
            {
                "role": "system",
                "content": "You answer questions about this document:"
                + "\n"
                + extracted_text,
            }]    
        st.session_state["number_tokens"] = []
        st.session_state["model_name"] = []
        st.session_state["cost"] = []
        st.session_state["total_cost"] = 0.0
        st.session_state["total_tokens"] = []
        # counter_placeholder.write(
        #     f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}"
        # )

    # generate a response


    # container for chat history
    response_container = st.container()
    # container for text box
    
                

         

    if st.session_state["generated"]:
        with response_container:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))
                st.write(
                    f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}"
                )
                # counter_placeholder.write(
                #     f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}"
                # )
