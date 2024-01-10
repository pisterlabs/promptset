import streamlit as st
import os
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.storage import LocalFileStore

st.set_page_config(
    "SiteGPT",
    "🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

llm = ChatOpenAI(
    temperature=0.1,
)


answer_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
"""
)


def get_answers(input):
    docs = input["docs"]
    question = input["question"]

    answer_chain = answer_prompt | llm

    # for doc in docs:
    #     result = answer_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    # st.write(answers)

    return {
        "question": question,
        "answers": [
            {
                "answer": answer_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(input):
    answers = input["answers"]
    question = input["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        [
            f"{answer['answer']}\n{answer['source']}\n{answer['date']}\n"
            for answer in answers
        ]
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    cache_dir = LocalFileStore(f"./.cache/embeddings/Site/{url}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(), cache_dir
    )
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()


def store_emb_history(question, answer):
    if "history_db" in st.session_state:
        vector_store = st.session_state["history_db"]
        vector_store.add_texts([question], metadatas=[{"answer": answer}])
    else:
        vector_store = FAISS.from_texts(
            [question], OpenAIEmbeddings(), metadatas=[{"answer": answer}]
        )
        st.session_state["history_db"] = vector_store
    vector_store.save_local("./.cache/embeddings/history")


def load_emb_history():
    # check directory is empty
    if os.path.exists("./.cache/embeddings/history") and os.listdir(
        "./.cache/embeddings/history"
    ):
        vector_store = FAISS.load_local(
            "./.cache/embeddings/history", OpenAIEmbeddings()
        )
        st.session_state["history_db"] = vector_store
        return True
    else:
        print("DB directory is empty")
        return False


def search_history(query):
    vector_store = st.session_state["history_db"]
    result = vector_store.similarity_search_with_score(query)
    return_list = []
    for text, score in result:
        if score < 0.1:
            return_list.append({"text": text, "score": score})

    return return_list


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_massage(message, role, save_flag):
    with st.chat_message(role):
        st.markdown(message)
    if save_flag:
        save_message(message=message, role=role)


def paint_messages():
    if "messages" in st.session_state:
        for message in st.session_state["messages"]:
            send_massage(
                message=message["message"], role=message["role"], save_flag=False
            )


with st.sidebar:
    url = st.text_input(
        "URL",
        placeholder="https://www.example.com",
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Write down a Sitempa URL.")
    else:
        retriever = load_website(url)
        paint_messages()

        message = st.chat_input("Ask a question of the website.")

        if message:
            send_massage(message, "human", True)
            history_db_flag = load_emb_history()
            keep_question = True
            if history_db_flag:
                similar_text_list = search_history(message)
                if len(similar_text_list) != 0:
                    history_message = "I found this in the history\n\n"
                    send_massage(history_message, "ai", False)
                    for similar_text in similar_text_list:
                        send_massage(
                            f"Question: {similar_text['text'].page_content}\n\nAnswer: {similar_text['text'].metadata}\n\nScore: {similar_text['score']}\n\n",
                            "system",
                            False,
                        )
                    keep_question = st.button("Keep your question?")
            if keep_question:
                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )

                result = chain.invoke(message)
                store_emb_history(message, result.content)
                send_massage(result.content, "ai", True)

        else:
            st.session_state["messages"] = []
