import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from qdrant_client import QdrantClient

from dotenv import load_dotenv
from langchain.vectorstores.qdrant import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from references.htmlTemplates import css, bot_template, user_template
from utils import get_pdf_text, get_text_chunks

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),  # type: ignore
    task_type="retrieval_document",
)

# db_client = QdrantClient(
#     url=os.getenv("QDRANT_URL"),
#     # api_key=os.getenv('QDRANT_API_KEY')
# )


def make_vectorstore(pdf_docs, collection_name):
    text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(text)

    vectors = Qdrant.from_texts(
        url=os.getenv("QDRANT_URL"),
        collection_name=collection_name,
        texts=text_chunks,
        embedding=embeddings,
        prefer_grpc=True,
        force_recreate=True,
    )
    print("from create:", vectors)

    return


def get_db_collections() -> dict:
    """get collections name and vector count"""
    client = QdrantClient(os.getenv("QDRANT_URL"))

    collections_list = [
        collection["name"]
        for collection in client.get_collections().model_dump()["collections"]
    ]

    info_list = [
        client.get_collection(collection_name=collection).model_dump()["vectors_count"]
        for collection in collections_list
    ]

    return {"collection_name": collections_list, "caption": info_list}


def disabled_get_conversation_chain(vectorstore: Qdrant):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        convert_system_message_to_human=True,
        # client=db_client,
    )  # type: ignore

    # load vector store from qdrant
    # prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    # {memory}

    # Question: {question}
    # Helpful Answer:"""
    # my_prompt = PromptTemplate(
    #     template=prompt_template, input_variables=[str(memory), "question"]
    # )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True,
    )
    return conversation_chain


def load_db_collection(collection) -> Qdrant:
    """load vector store from qdrant"""
    client = QdrantClient(os.getenv("QDRANT_URL"))

    return Qdrant(client=client, collection_name=collection, embeddings=embeddings)


def handle_userinput(user_question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        convert_system_message_to_human=True,
        # client=db_client,
    )  # type: ignore
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(),
        memory=memory,
    )

    response = conversation_chain({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "collection" not in st.session_state:
        st.session_state.collection = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    with st.sidebar:
        with st.container(border=True):
            # st.subheader("你的资料库")

            db_list = get_db_collections()
            db_selection = st.radio(
                label="选择一个资料库",
                options=db_list["collection_name"],
                captions=[f"vectors:{n}" for n in db_list["caption"]],
                horizontal=True,
            )
            # print(db_selection)

            if st.button("选择"):
                with st.spinner("读取中..."):
                    # create conversation chain
                    st.session_state.collection = db_selection
                    st.session_state.vectorstore = load_db_collection(db_selection)
                    # print("from load=", vectorstore)

        with st.container(border=True):
            # st.subheader("Your documents")
            collection_name = st.text_input(
                label="资料库名称",
                help="指定数据库collection name",
                placeholder="research_1",
            )
            pdf_docs = st.file_uploader(
                "上传PDF文件",
                accept_multiple_files=True,
                type=["pdf"],
            )

            if collection_name and pdf_docs:
                if st.button("处理"):
                    with st.spinner("处理中..."):
                        # create vector store
                        make_vectorstore(pdf_docs, collection_name)
                        st.rerun()

    st.header(
        f"Chat with {st.session_state.collection if st.session_state.collection else 'your PDFs'} :books:"
    )
    if st.session_state.collection:
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)
    else:
        st.write("上传或者选择资料库")


if __name__ == "__main__":
    main()
