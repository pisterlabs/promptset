import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import openai
from langchain.document_loaders import TextLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import CharacterTextSplitter, Language, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from htmlTemplates import bot_template, user_template, css
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import os
import read_gitlink


def get_content_from_files(directory,filename):
    # Initialize an empty string to store the concatenated text
    content = ""
    ## if passing individual files 
    if filename and os.path.isfile(os.path.join(directory,filename)):
        with open(os.path.join(directory, filename), "r") as file_obj:
            content += file_obj.read()
    else:

        # Check if directory is valid
        if not os.path.exists(directory):
            raise ValueError(f"Directory {directory} does not exist.")

        # Get a list of text files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        for file in files:
            with open(os.path.join(directory, file), "r") as file_obj:
                content += file_obj.read()

    return content


def get_code_file(filename):
    # The base_path should be the path to the directory where your code files are located.
    loader = GenericLoader.from_filesystem(
        "codes/"+filename,  # Base directory where the code files are stored
        glob="**/*",  # Glob pattern to recursively search for files
        # If the loader supports loading all file types without specifying suffixes, you can comment out or remove the line below.
        # If you still need to specify suffixes, list all the file extensions you're interested in.
        suffixes=[".cpp", ".h", ".py", ".java", ".txt", ".md", "..."],  # Add all file extensions you want to include
        parser=LanguageParser(parser_threshold=500)  # Use AUTO if the parser can automatically detect the language, otherwise, you may need separate parsers for each language type
    )
    files = loader.load()

    return files

def get_all_files_loader():
    # The base_path should be the path to the directory where your code files are located.
    loader = GenericLoader.from_filesystem(
        "codes",  # Base directory where the code files are stored
        glob="**/*",  # Glob pattern to recursively search for files
        # If the loader supports loading all file types without specifying suffixes, you can comment out or remove the line below.
        # If you still need to specify suffixes, list all the file extensions you're interested in.
        suffixes=[".cpp", ".h", ".py", ".java", ".txt", ".md", "..."],  # Add all file extensions you want to include
        parser=LanguageParser(parser_threshold=500)  # Use AUTO if the parser can automatically detect the language, otherwise, you may need separate parsers for each language type
    )
    #return loader.load()  # Load the files and return the parsed data

    files = loader.load()  # This may not be the correct method call depending on the langchain implementation.

    return files 

def get_file_loader(filename):
    # The base_path should be the path to the directory where your code files are located.
    loader = GenericLoader.from_filesystem(
        "codes/",  # Base directory where the code files are stored
        glob= '**/*/' + filename,  # Glob pattern to recursively search for files
        # If the loader supports loading all file types without specifying suffixes, you can comment out or remove the line below.
        # If you still need to specify suffixes, list all the file extensions you're interested in.
        suffixes=[".cpp", ".h", ".py", ".java", ".txt", ".md", "..."],  # Add all file extensions you want to include
        parser=LanguageParser(language=Language.CPP, parser_threshold=500)  # Use AUTO if the parser can automatically detect the language, otherwise, you may need separate parsers for each language type
    )
    #return loader.load()  # Load the files and return the parsed data

    files = loader.load()  # This may not be the correct method call depending on the langchain implementation.

    return files 


def summarise_file():
    prompt_template = """generate technical documentation for a junior software engineer for the below code base by giving a technical details for each file independently:
            "{text}"
            :"""
    prompt = PromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    docs = get_all_files_loader()

    print(docs)
    print("waiting")
    summary_text = stuff_chain.run(docs)
    print("done")
    f = open("summary/summary.txt", "a+")
    f.write(summary_text)
    f.close()

    return docs

def get_code_chunks(code_text):
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                   chunk_size=50, chunk_overlap=0)
    python_docs = python_splitter.create_documents([str(code_text)])
    print('document:',python_docs[0])
    return python_docs

def get_text_chunks(raw_text):
    ## create a new instance 
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)

    chunks = text_splitter.split_text(raw_text)  ## returns a list of chunks with each chunk size 100

    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    ## initialize the conversation chain
    converstation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return converstation_chain


def handle_user_input(user_question, user):
    prompt = f"Explain the query {user_question} to the user who is has a {user} background"
    response = st.session_state.conversation({'question': prompt})

    # Update the chat history
    st.session_state.chat_history = response['chat_history']

    # Add the response to the UI
    for i, message in enumerate(st.session_state.chat_history):
        # Check if the message is from the user or the chatbot
        if i % 2 == 0:
            # User message
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            # Chatbot message
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    ## this allows langchain access to the access tokens.Since we are using langchain , follow the same variable format
    load_dotenv()

    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Enterprise CodeBuddy! 🤖")

    #st.sidebar.success("Select a Repo below.")

    st.markdown(
        """
        CodeBuddy can help you

        1. Generate technical documentation for your repo 
        2. Chat with repo
        3. Streamline Knowledge Transfer
        
        """)
    st.write(css, unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

   
    

    with st.sidebar:
        if st.button("Update DB"):
            with st.spinner("Processing"):

                raw_text = get_content_from_files('summary/')

                ## get the text chunks
                text_chunks = get_text_chunks(raw_text)
                print('length:',type(text_chunks[0]))

                ## create vector store
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
