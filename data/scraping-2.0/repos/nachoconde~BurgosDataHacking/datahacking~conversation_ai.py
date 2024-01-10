import os
import constants
from operator import itemgetter
from typing import Sequence
from functools import partial
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from utils import DocsJSONLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import SQLRecordManager, index
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import (
    AIMessage,
    BaseRetriever,
    Document,
    HumanMessage,
    StrOutputParser,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.document_transformers import LongContextReorder
from langchain.schema.messages import BaseMessageChunk
from langchain.schema.runnable import Runnable, RunnableMap
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from prompt_templates import CONDENSE_QUESTION_TEMPLATE, SYSTEM_ANSWER_QUESTION_TEMPLATE
from utils import num_tokens_from_string, load_config

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY


def load_doc_and_split(file_path: str):
    loader = DocsJSONLLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, length_function=num_tokens_from_string, chunk_overlap=50
    )

    splitted_docs = text_splitter.split_documents(data)
    return splitted_docs


def create_retriever_chain(
    llm,
    retriever: BaseRetriever,
    use_chat_history: bool,
):
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)

    if not use_chat_history:
        initial_chain = (itemgetter("question")) | retriever
        return initial_chain
    else:
        condense_question_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser()
        )
        conversation_chain = condense_question_chain | retriever
        return conversation_chain


def get_k_or_less_documents(documents: list[Document], k: int):
    if len(documents) <= k:
        return documents
    else:
        return documents[:k]


def reorder_documents(documents: list[Document]):
    reorder = LongContextReorder()
    return reorder.transform_documents(documents)


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs: list[str] = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def create_answer_chain(
    llm,
    retriever: BaseRetriever,
    use_chat_history: bool,
    k: int = 5,
) -> Runnable:
    retriever_chain = create_retriever_chain(llm, retriever, use_chat_history)

    _get_k_or_less_documents = partial(get_k_or_less_documents, k=k)

    context = RunnableMap(
        {
            "context": (
                retriever_chain
                | _get_k_or_less_documents
                | reorder_documents
                | format_docs
            ),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    )

    prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", SYSTEM_ANSWER_QUESTION_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = prompt | llm | StrOutputParser()
    response_chain = context | response_synthesizer

    return response_chain


db_path = "datahacking/db/chroma_db"
config = load_config()
documents = []

jsonl_files = [
    section.get("jsonl_file")
    for section in config.values()
    if section.get("jsonl_file")
]
for jsonl_file in jsonl_files:
    documents += load_doc_and_split(jsonl_file)


print("Docs Loaded")

record_manager = SQLRecordManager(namespace="burgos", db_url="sqlite:///_memory:")
record_manager.create_schema()
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = Chroma(
    collection_name="burgos",
    persist_directory=db_path,
    embedding_function=embeddings,
)
print("Indexing")
indexing_result = index(
    docs_source=documents,
    record_manager=record_manager,
    vector_store=vectorstore,
    cleanup="full",
    batch_size=1000,
)

print("indexing_result", indexing_result)

vector_keys = vectorstore.get(
    ids=record_manager.list_keys(), include=["documents", "metadatas"]
)

docs_in_vectorstore = [
    Document(page_content=page_content, metadata=metadata)
    for page_content, metadata in zip(
        vector_keys["documents"], vector_keys["metadatas"]
    )
]
print("Creating retrievers")
keyword_retriever = BM25Retriever.from_documents(docs_in_vectorstore)
keyword_retriever.k = 6


semantic_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20,
        "lambda_mult": 0.3,
    },
)
print("Semantic retrievers ready")


queries_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=semantic_retriever,
    llm=queries_llm,
)

retriever = EnsembleRetriever(
    retrievers=[keyword_retriever, multi_query_retriever],
    weights=[0.3, 0.7],
    c=0,
)

st.title("Hackeando los datos de Burgos!")

st.subheader(
    "Usamos una combinacion de datos de fuentes abiertas para resolver preguntas sobre economía, industria y empleo en Burgos."
)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Pregunta aqui cosas sobre la economía de Burgos"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Create answer chain
        # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.0)

        use_chat_history = len(st.session_state.messages) > 1

        chat_history = []
        if use_chat_history:
            for message in st.session_state.messages[:-1]:
                if message["role"] == "user":
                    chat_history.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    chat_history.append(AIMessage(content=message["content"]))

        answer_chain = create_answer_chain(
            llm=llm,
            retriever=retriever,
            use_chat_history=use_chat_history,
            k=6,
        )
        # answer_chain = create_answer_chain(
        #     llm=llm, retriever=retriever, use_chat_history=False, k=6
        # )

        message_placeholder = st.empty()
        full_response = ""
        for token in answer_chain.stream(
            {
                "question": prompt,
                "chat_history": chat_history,
            }
        ):
            full_response += token
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
