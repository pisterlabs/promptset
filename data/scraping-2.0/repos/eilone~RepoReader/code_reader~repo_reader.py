### Imports
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from openai.error import InvalidRequestError
from chromadb.errors import NoIndexException, NotEnoughElementsException
import os

from .reader_utils import (
        get_llm_api,
        clone_repo,
        pull_repo,
        create_vectordb,
        reset_history,
        format_answer,
        format_question,
        format_exception,
        show_sources,
        chat_with_llm_model,
        print_color,
        get_openai_api_key,
        )
from .reader_config import (
        LOCAL_PATH,
        LLM_TEMPERATURE,
        LLM_MODEL_NAME,
        PURPLE,
        GREEN,
        GREY,
        NUM_SOURCE_DOCS,
        is_reset_history,
        github_url as GITHUB_URL,
        )
from code_reader.llm_input import context_template
import streamlit as st

### Main


def repo_reader(repo_url, num_src_docs):
    if is_reset_history: reset_history()

    ### Clone repo from Github
    repo_name, is_repo_cloned, *_ = clone_repo(repo_url)

    slider = st.slider(
        label='Num of Relevant Docs Input', min_value=1,
        max_value=30, value=num_src_docs, key='docs_slider')

    num_src_docs = slider

    persist_directory = f'chroma_db/chroma_db_{repo_name}'
    embedding = OpenAIEmbeddings()

    if (not os.path.exists(persist_directory)) or (not (is_repo_cloned)) or st.session_state['HARD_RESET_DB']:
        create_vectordb(LOCAL_PATH, repo_name, embedding, persist_directory)

    # Now we can load the persisted database from disk, and use it as normal
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)
    print('Called existing VectorDB')

    ### Create a retriever from the indexed DB
    retriever = vectordb.as_retriever(search_kwargs={"k": num_src_docs})

    st.write("Ask a question about the repository, BE SPECIFIC:")
    ## Check the retriever
    try:
        docs = retriever.get_relevant_documents("What is the name of the repo?")
        print(len(docs))
    except Exception as e:
        if isinstance(e, NoIndexException):
            create_vectordb(LOCAL_PATH, repo_name, embedding, persist_directory)
        elif isinstance(e, NotEnoughElementsException):
            err_msg = f"=== Try reducing the 'Relevant Docs' slider (currently {num_src_docs}) ===\n"
            st.error(e.__str__())
            st.write(
                format_exception(err_msg),
                unsafe_allow_html=True)

    ## Call the LLM API
    turbo_llm = get_llm_api(LLM_MODEL_NAME, LLM_TEMPERATURE)

    ### Integrate LLM API and source-docs in the Chain
    get_openai_api_key()
    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True,
                                           )

    ### START CHATTING!

    conversation = []
    # This will hold the current query. Whenever the user submits a new query, it gets added here.
    query = st.text_input("Your question:")

    # is query changed?
    is_query_changed = st.session_state['last_q'] != query

    # Check if the user has entered a new query.
    if query and is_query_changed:
        try:
            st.session_state['last_q'] = query
            st.session_state['conversation_history'] += f'Last Question: {query} \n'

            print_color(f"QUESTION\n\n{query}", PURPLE)

            # Perform the chat operation.
            result, sources = chat_with_llm_model(query, qa_chain, repo_name, repo_url, context_template)

            st.session_state['conversation_history'] += f'\nLast Answer: {result}\n\n'
            print_color(f"ANSWER\n\n{result}", GREEN)



            # Add this interaction to the conversation history.
            st.session_state.conversation.append((query, result, sources))

            # Display the question, answer, and sources.
            st.write(format_question(f"Question: {query}"), unsafe_allow_html=True)
            st.write(format_answer(f"Answer: {result}"), unsafe_allow_html=True)


            # show sources in sidebar
            show_sources(sources)

            print_color(f"SOURCES:", GREY)
            for src in sources:
                print_color(src, GREY)

        except Exception as e:
            st.write(format_exception(e), unsafe_allow_html=True)
            if isinstance(e, InvalidRequestError) or isinstance(e, NotEnoughElementsException):
                err_msg = f"=== Try reducing the 'Relevant Docs' slider (currently {num_src_docs}) ===\n"
                st.write(
                    format_exception(err_msg),
                    unsafe_allow_html=True)

    # Show history
    if st.session_state['show_history']:
        st.subheader("Conversation History")
        st.write(st.session_state['conversation_history'], unsafe_allow_html=True)

    # pull repo
    if st.session_state['pull_repo_btn']:
        pull_repo(GITHUB_URL, LOCAL_PATH)

    print('len conversation:', len(conversation))


def main():


    st.title("Repo Reader")
    st.subheader("Chat with your Repo!")
    st.sidebar.title("Sources")

    # Get the session state for this session.
    st.session_state['conversation'] = []

    # Prompt vars
    # ============#

    if 'conversation_history' not in st.session_state.keys():
        st.session_state['conversation_history'] = ""

    if 'last_q' not in st.session_state.keys():
        st.session_state['last_q'] = ""

    input_url = st.text_input("GitHub URL", GITHUB_URL)
    github_url = input_url

    start_col, history_col, reset_col, pull_repo_col = st.columns(4)

    st.session_state['start_btn'] = start_col.checkbox("Start Chatting")
    st.session_state['show_history'] = history_col.checkbox("Show History", value=False)
    st.session_state['HARD_RESET_DB'] = reset_col.button("Reset Chroma DB?")
    st.session_state['pull_repo_btn'] = pull_repo_col.button("Pull Repo")


    if st.session_state['start_btn']:
        repo_reader(github_url, NUM_SOURCE_DOCS)

if __name__ == "__main__":
    main()