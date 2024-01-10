import streamlit as st
import dotenv
import langchain
import json

from cassandra.cluster import Session
from cassandra.query import PreparedStatement

from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import BaseRetriever, Document, SystemMessage

from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider

# Enable langchain debug mode
langchain.debug = True

dotenv.load_dotenv(dotenv.find_dotenv())


class AstraProductRetriever(BaseRetriever):
    session: Session
    embedding: OpenAIEmbeddings
    lang: str = "English"
    search_statement_en: PreparedStatement = None
    search_statement_th: PreparedStatement = None

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query):
        docs = []
        embeddingvector = self.embedding.embed_query(query)
        if self.lang == "Thai":
            if self.search_statement_th is None:
                self.search_statement_th = self.session.prepare("""
                    SELECT
                        product_id,
                        brand,
                        saleprice,
                        product_categories,
                        product_name,
                        short_description,
                        long_description
                    FROM hybridretail.products_cg_hybrid
                    ORDER BY openai_description_embedding_th ANN OF ?
                    LIMIT ?
                    """)
            query = self.search_statement_th
        else:
            if self.search_statement_en is None:
                self.search_statement_en = self.session.prepare("""
                    SELECT
                        product_id,
                        brand,
                        saleprice,
                        product_categories,
                        product_name_en,
                        short_description_en,
                        long_description_en
                    FROM hybridretail.products_cg_hybrid
                    ORDER BY openai_description_embedding_en ANN OF ?
                    LIMIT ?
                    """)
            query = self.search_statement_en
        results = self.session.execute(query, [embeddingvector, 5])
        top_products = results._current_rows
        for r in top_products:
            docs.append(Document(
                id=r.product_id,
                page_content=r.product_name if self.lang == "Thai" else r.product_name_en,
                metadata={"product id": r.product_id,
                          "brand": r.brand,
                          "product category": r.product_categories,
                          "product name": r.product_name if self.lang == "Thai" else r.product_name_en,
                          "description": r.short_description if self.lang == "Thai" else r.short_description_en,
                          "price": r.saleprice
                          }
            ))

        return docs


def get_session(scb: str, secrets: str) -> Session:
    """
    Connect to Astra DB using secure connect bundle and credentials.

    Parameters
    ----------
    scb : str
        Path to secure connect bundle.
    secrets : str
        Path to credentials.
    """

    cloud_config = {
        'secure_connect_bundle': scb
    }

    with open(secrets) as f:
        secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    return cluster.connect()


@st.cache_resource
def create_chatbot(lang: str):
    print(f"Creating chatbot for {lang}...")
    session = get_session(scb='./config/secure-connect-multilingual.zip',
                          secrets='./config/multilingual-token.json')
    llm = ChatOpenAI(temperature=0, streaming=True)
    embedding = OpenAIEmbeddings()
    retriever = AstraProductRetriever(
        session=session, embedding=embedding, lang=lang)
    retriever_tool = create_retriever_tool(
        retriever, "products_retrevier", "Useful when searching for products from a product description. Prices are in THB.")
    system_message = "You are a customer service of a home improvement store and you are asked to pick products for a customer."
    if lang == "Thai":
        system_message = f"{system_message} All the responses should be in Thai language."
    message = SystemMessage(content=system_message)
    agent_executor = create_conversational_retrieval_agent(
        llm=llm, tools=[retriever_tool], system_message=message, verbose=True)
    return agent_executor


if 'history' not in st.session_state:
    st.session_state['history'] = {
        "English": [],
        "Thai": []
    }

st.set_page_config(layout="wide")

with st.sidebar:
    lang = st.radio(
        "Chat language",
        ["English", "Thai"],
        captions=[".", "Experimental"])

chatbot = create_chatbot(lang)


# Display chat messages from history on app rerun
for (query, answer) in st.session_state['history'][lang]:
    with st.chat_message("User"):
        st.markdown(query)
    with st.chat_message("Bot"):
        st.markdown(answer)

prompt = st.chat_input(placeholder="Ask chatbot")
if prompt:
    # Display user message in chat message container
    with st.chat_message("User"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("Bot"):
        st_callback = StreamlitCallbackHandler(st.container())
        result = result = chatbot.invoke({
            "input": prompt,
            "chat_history": st.session_state['history'][lang]
        }, config={"callbacks": [st_callback]})
        st.session_state['history'][lang].append((prompt, result["output"]))
        st.markdown(result["output"])