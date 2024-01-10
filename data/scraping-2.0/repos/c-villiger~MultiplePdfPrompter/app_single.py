import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os
import tempfile

# Chains
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import langchain
from termcolor import colored


def get_text(pdf_docs, k, max_tokens=4096):
    chunk_size = int(max_tokens/k)-1
    all_pages = []

    for pdf in pdf_docs:
        # Create a temporary file to store the uploaded PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(pdf.getvalue())
        temp_file_path = temp_file.name
        temp_file.close()

        loader = PyPDFLoader(temp_file_path)
        file = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_size/2)
        pages = text_splitter.split_documents(file)

        # Add metadata to each page
        for page in pages:
            page.metadata["source"] = os.path.basename(pdf.name)
            all_pages.append(page)

        all_pages.extend(pages)

        # Cleanup: Remove the temporary file
        os.remove(temp_file_path)

    return all_pages


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # Extract page_content from each dictionary in text_chunks

    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore


def conversation_chain(vectorstore, chain_type, k, own_knowledge=False, show_pages=False):

    # Import API Key
    from apikey import API_KEY
    os.environ["OPENAI_API_KEY"] = API_KEY

    # =========== #
    # Prompts
    # =========== #

    # Define Chain
    if own_knowledge:
        prompt_template = """Use the following pieces of chat history and context to answer the question at the end. \
            If the answer does not become clear from the context, you can also use your own knowledge. \
            If you use your own knowledge, please indicate this clearly in your answer. \

        Context:
        {context}

        {question}
        Helpful answer:"""

    if not own_knowledge:

        prompt_template = """Use the following pieces of chat history and context to answer the question at the end. \
            Do NOT use your own knowledge and give the best possible answer from the context.\
        
        Context:
        {context}

        {question}
        Helpful answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    # =========== #
    # Chains
    # =========== #
    # QA chain that is adaptable
    # Amount of returned documents k-i -> makes it adaptable. Otherwise, it would always return k documents and the output would be the same.
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k})

    # Define retrieval chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    return qa


def return_sources(result):
    sources = [(doc.page_content, os.path.basename(doc.metadata["source"]), f"page: {doc.metadata['page']}") for doc in
               result['source_documents']]

    print("\n--------------------------------------------------------------------")
    print(colored("Sources:", "green"))
    print("--------------------------------------------------------------------\n")

    # Print Sources and Sort them first
    for source_key in sources:
        print(colored(f"Source: {source_key[0]} {source_key[1]}", "red"))
        print("\n")


def return_answer(result):
    answer = result['result']
    return answer


def handle_userinput(user_question):
    result = st.session_state.conversation(
        user_question)

    st.write(
        "--------------------------------------------------------------------")
    st.markdown(
        f"**Answer:**", unsafe_allow_html=True)
    st.write(
        "--------------------------------------------------------------------")
    st.write(return_answer(result))
    st.write(
        "--------------------------------------------------------------------")
    st.markdown(
        f"**Sources:**", unsafe_allow_html=True)
    st.write(
        "--------------------------------------------------------------------")

    # Display Sources and Sort them first
    for i, source_key in enumerate(result['source_documents']):
        st.markdown(
            f'<span style="color:red">**Source {i+1}:** {source_key.metadata["source"]}, page: {source_key.metadata["page"]}</span>',
            unsafe_allow_html=True
        )
        st.markdown(
            f"{source_key.page_content}", unsafe_allow_html=True)
        st.write("\n")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:

        # Model specifications
        st.subheader("Model specifications")
        chain_type = st.selectbox(
            'Chain type:',
            ('stuff', 'refine', 'map reduce', 'map re-rank'))
        k = st.slider('Number of sources:', 0, 15, 1)

        # Your documents
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):

                # get pdf text
                text_chunks = get_text(pdf_docs, k)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = conversation_chain(
                    vectorstore, chain_type, k, own_knowledge=False, show_pages=False)

                st.write("Done! You can now ask a question about your documents.")


if __name__ == '__main__':
    main()
