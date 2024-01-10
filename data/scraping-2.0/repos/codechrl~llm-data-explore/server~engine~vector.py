import json
import os
import pathlib
import pickle
import re
import subprocess

import tiktoken
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# from langchain.prompts import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     MessagesPlaceholder,
# )
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from setting import setting

os.environ["OPENAI_API_KEY"] = setting.OPENAI_API_KEY

REPO_URL = "https://github.com/GovTechSG/developer.gov.sg"  # Source URL
DOCS_FOLDER = "data/repository"  # Folder to check out to
REPO_DOCUMENTS_PATH = ""  # Set to "" to index the whole data folder
DOCUMENT_BASE_URL = "https://www.developer.tech.gov.sg/products/categories/devops/ship-hats"  # Actual URL
DATA_STORE_DIR = "data/data_store"

name_filter = "**/*.md"
name_filter = "**/*.*"
separator = "\n### "
separator = " "  # Thi+s separator assumes Markdown docs from the repo uses ### as logical main header most of the time
name_filter = "**/*.*"
separator = "\n### "
separator = " "  # Thi+s separator assumes Markdown docs from the repo uses ### as logical main header most of the time
chunk_size_limit = 1000
max_chunk_overlap = 20

CONV_LIST = []


def run_command_with_output(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True
    )

    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())

    return process.poll()


def convert_path_to_doc_url(doc_path):
    # Convert from relative path to actual document url
    return re.sub(
        f"{DOCS_FOLDER}/{REPO_DOCUMENTS_PATH}/(.*)\.[\w\d]+",
        f"{DOCUMENT_BASE_URL}/\\1",
        str(doc_path),
    )


def split_doc(title, folder="repository", only=None):
    repo_path = pathlib.Path(os.path.join(f"data/{folder}", title))
    # document_files = list(repo_path.glob(name_filter))


def split_doc(title, folder="repository", only=None):
    repo_path = pathlib.Path(os.path.join(f"data/{folder}", title))
    # document_files = list(repo_path.glob(name_filter))
    document_files = list(repo_path.glob(name_filter))
    document_files = [p for p in document_files if os.path.isfile(p)]
    document_files = [
        p
        for p in document_files
        if str(p).split(".")[-1] not in ["jpg", "jpeg", "mp3", "mp4", "png", "webp"]
    ]

    document_files = [p for p in document_files if "/.git/" not in str(p)]

    if only:
        document_files = [p for p in document_files if str(p).split(".")[-1] in only]

    documents = []
    for file in document_files:
        try:
            print(file)
            documents.append(
                Document(
                    page_content=open(file, "r").read(),
                    metadata={"source": convert_path_to_doc_url(file)},
                )
            )
        except:
            pass

    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size_limit,
        chunk_overlap=max_chunk_overlap,
    )
    split_docs = text_splitter.split_documents(documents)

    enc = tiktoken.get_encoding("cl100k_base")
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
    except:
        pass
    enc = tiktoken.get_encoding("cl100k_base")
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
    except:
        pass

    total_word_count = sum(len(doc.page_content.split()) for doc in split_docs)
    total_token_count = sum(len(enc.encode(doc.page_content)) for doc in split_docs)

    print(f"\nTotal word count: {total_word_count}")
    print(f"\nEstimated tokens: {total_token_count}")
    print(f"\nEstimated cost of embedding: ${total_token_count * 0.0004 / 1000}")

    return split_docs


def generate_embedding(title, split_docs):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(f"{DATA_STORE_DIR}/{title}")


def load_embedding(title):
    if os.path.exists(f"{DATA_STORE_DIR}/{title}"):
        vector_store = FAISS.load_local(f"{DATA_STORE_DIR}/{title}", OpenAIEmbeddings())
        return vector_store
    else:
        print(
            f"Missing files. Upload index.faiss and index.pkl files to {DATA_STORE_DIR}/{title} directory first"
        )


def ask(vector_store, question, stream=False):
    system_template = """Use the following pieces of context to answer the users question.
    Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
    If you don't know the answer, just say that "I don't know", don't try to make up an answer.
    
    ----------------
    {summaries}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k", temperature=0, max_tokens=14000
    )  # Modify model_name if you have access to GPT-4
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    if not stream:
        return chain(question)
    else:
        return chain.stream(question)


def ask_conv(vector_store, question, conv_id, stream=False):
    global CONV_LIST
    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation

    try:
        chat_history = [ch["chat_history"] for ch in CONV_LIST if ch["id"] == conv_id]
    except:
        chat_history = []
        CONV_LIST.append({"id": conv_id, "chat_history": []})

    llm = OpenAI(
        temperature=0,
    )
    streaming_llm = OpenAI(
        # streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0,
    )

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

    qa = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
    )
    chat_history = []
    result = qa({"question": question, "chat_history": chat_history})

    chat_history.append((question, result["answer"]))

    for idx in range(len(CONV_LIST)):
        if CONV_LIST[idx]["id"] == conv_id:
            CONV_LIST[idx]["chat_history"] = chat_history

    return result


def ask_memory_(vector_store, question, session_id, stream=False):
    try:
        with open(f"data/session/{session_id}.json", "r") as json_file:
            json.load(json_file)
    except:
        pass

    system_template = """Use the following pieces of context to answer the users question.
    Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
    If you don't know the answer, just say that "I don't know", don't try to make up an answer.
    
    ----------------
    {summaries}"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        # MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, max_tokens=14000)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        # memory=memory,
    )
    try:
        print(memory.to_json())
    except:
        pass

    if not stream:
        return chain(question)
    else:
        return chain.stream(question)


def ask_memory(vector_store, question, session_id=None, stream=False):
    try:
        with open(f"data/session/{session_id}.pickle", "rb") as file:
            memory = pickle.load(file)
    except:
        memory = []

    memory_template = (
        """Also use the following pieces of chat history to understand context."""
    )
    if memory:
        for i_memory in memory:
            memory_template += f"{i_memory[0]}: {i_memory[1]}"

    system_template = """Use the following pieces of context to answer the users question.
    Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
    If you don't know the answer, just say that "I don't know", don't try to make up an answer.
    
    ----------------
    {summaries}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(memory_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, max_tokens=14000)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    result = chain(question)

    try:
        memory.append(["Human", question])
        memory.append(["Human", result["answer"]])
        print(memory)
        with open(f"data/session/{session_id}.pickle", "wb") as file:
            pickle.dump(memory, file)
    except Exception as exc:
        print(exc)

    return result


def repository_overview():
    repo_done = []
    directory_path = "data/db"
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            with open(file_path, "r") as file:
                json_data = json.load(file)
                if json_data.get("status") == "done":
                    repo_done.append(json_data)
    print(repo_done)
    model_embedding = load_embedding("overview_repository")
    if model_embedding is None:
        c = 0
        while True:
            try:
                model_embedding = load_embedding(repo_done[c]["title"])
                c += 1
                break
            except:
                pass

    for repo in repo_done:
        try:
            load_embedding(repo["title"])
            print(f"embed {repo['title']}")
            split_docs = split_doc(repo["title"], folder="raw", only=["txt"])
            # embeddings = OpenAIEmbeddings()
            # vector_store = FAISS.from_documents(split_docs, embeddings)
            # model_embedding.add_documents(split_docs)
            model_embedding.aadd_texts(split_docs)
            # model_embedding.a
            # model_embedding.add(model_repo.as_retriever())
            # model_embedding += model_repo
            # model_embedding = FAISS.IndexIDMap(FAISS.IndexFlatIP(faiss_index1.index.d))
            # model_embedding.add_with_ids(faiss_index1.index, np.arange(faiss_index1.index.ntotal))
            # model_embedding.add_with_ids(faiss_index2.index, np.arange(faiss_index1.index.ntotal, faiss_index1.index.ntotal + faiss_index2.index.ntotal))
        except:
            pass
    model_embedding.save_local(f"{DATA_STORE_DIR}/overview_repository")
