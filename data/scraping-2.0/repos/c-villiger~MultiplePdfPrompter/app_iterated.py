import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
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
from langchain.chains.summarize import load_summarize_chain
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
            chunk_size=chunk_size, chunk_overlap=int(chunk_size/2))
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


def conversation_chain(user_prompt, list_vectorstore, chain_type, own_knowledge=False, show_pages=False):

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

    # Define summary chain
    text_splitter = CharacterTextSplitter()
    qa_condense = load_summarize_chain(
        llm=OpenAI(temperature=0), chain_type="stuff")

    extended_answers = []
    unique_sources = {}

    for num_chunks, vectorstore in list_vectorstore.items():
        # QA chain that is adaptable
        # Amount of returned documents k-i -> makes it adaptable. Otherwise, it would always return k documents and the output would be the same.
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": num_chunks})

        # Define retrieval chain
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        # Run Chain with parameters
        result = qa(user_prompt)

        # Get Sources
        unique_sources[num_chunks] = (result['source_documents'])

        # Append result to extended_answers
        extended_answers.append(result['result'])

    # ========== #
    # Final dfs
    # ========== #

    # Combine extended_answers
    combined_result = ' '.join(extended_answers)

    # Run the qa function on the combined_result (summary)
    texts = text_splitter.split_text(combined_result)
    docs = [Document(page_content=t) for t in texts[:3]]

    condensed_result = str(qa_condense.run(docs))

    return combined_result, condensed_result, unique_sources


def handle_userinput(user_prompt, list_vectorstore):

    # create conversation chain
    combined_result, condensed_result, combined_sources = conversation_chain(
        user_prompt, list_vectorstore=st.session_state.list_vectorstore, chain_type=st.session_state.chain_type, own_knowledge=False, show_pages=False)

    st.write(
        "--------------------------------------------------------------------")
    st.markdown(
        f"**Combined answer:**", unsafe_allow_html=True)
    st.write(
        "--------------------------------------------------------------------")
    st.write(combined_result)
    st.write(
        "--------------------------------------------------------------------")
    st.markdown(
        f"**Condensed answer:**", unsafe_allow_html=True)
    st.write(
        "--------------------------------------------------------------------")
    st.write(condensed_result)
    st.write(
        "--------------------------------------------------------------------")
    st.markdown(
        f"**Sources:**", unsafe_allow_html=True)
    st.write(
        "--------------------------------------------------------------------")

    # Create an empty set to store displayed sources
    displayed_sources = set()

    count = 1

    # Display Sources and ensure they are unique
    for i, source_key in combined_sources.items():

        for j in range(len(source_key)):
            source_info = (source_key[j].metadata["source"],
                           source_key[j].metadata["page"])

            # Check if the source has already been displayed
            if source_info not in displayed_sources:
                st.markdown(
                    f'<span style="color:red">**Source {count}:** {source_key[j].metadata["source"]}, page: {source_key[j].metadata["page"]}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"{source_key[j].page_content}", unsafe_allow_html=True)
                st.write("\n")

                # Add the source to the set of displayed sources
                displayed_sources.add(source_info)

                count += 1


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
        handle_userinput(user_question, st.session_state.list_vectorstore)

    with st.sidebar:
        with st.expander("Description"):
            st.write(
                "For details on the chain type, refers to the langchain website.")
            st.write("Number of sources per document refers to the number of similar chunks that are chosen per document. Eventually, only unique sources are displayed, \
                    meaning that text chunks from the same source are not displayed twice.")
            st.write("Number of iteration refers to the number of times the prompt is run. The number of sources per document is decreased by 1 every iteration to generate smaller chunks \
                    and eventually a more condensed answer. The number of iterations cannot be bigger than the number of sources per document.")

        # Model specifications
        st.subheader("Model specifications")
        st.session_state.chain_type = st.selectbox(
            'Chain type:',
            ('stuff', 'refine', 'map reduce', 'map re-rank'))
        num_chunks = st.slider('Number of sources per document:', 0, 15, 1)
        num_iterations = st.slider(
            'Number of iterations (cannot be bigger than number of sources per document!):', 0, 15, 1)

        # Your documents
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):

                list_vectorstore = {}
                for i in range(num_iterations):
                    # Change number of chunks every time
                    k = num_chunks-i

                    # get pdf text
                    text_chunks = get_text(pdf_docs, k)

                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Append vectorstore to list
                    list_vectorstore[k] = vectorstore

                st.session_state.list_vectorstore = list_vectorstore

                st.write("Done! You can now ask a question about your documents.")


if __name__ == '__main__':
    main()
