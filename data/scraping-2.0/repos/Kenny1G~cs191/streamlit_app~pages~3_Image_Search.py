import logging
import streamlit as st
from streamlit_image_select import image_select
from langchain.llms import OpenAI
from huggingface_hub import InferenceClient
from langchain.retrievers import RePhraseQueryRetriever
from langchain_community.vectorstores import Pinecone
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from utils import load_embedder, get_pinecone_image_index

logging.basicConfig()
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.DEBUG)

if "credentials" not in st.session_state or "uid" not in st.session_state:
    st.warning(
        "You are not authenticated yet. Please enter your unique ID in Setup Demo to Authenticate."
    )
    st.stop()

if "media_items_df" not in st.session_state:
    st.warning(
        "No media items in session, Please go to the Upsert Images page to upload images."
    )
    st.stop()

im_index_name = "photo-captions"
uid = st.session_state["uid"]
media_items_df = st.session_state["media_items_df"]

month_names = [
    "NOOP",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


AGENT_QUERY = """Prompt: You are an assistant tasked with taking a series of \
natural language queries from a user and synthesizing them into a single query \
for a vectorstore of CLIP embeddings. In this process, you strip out information \
that is not relevant for the retrieval task and focus on visual elements and \
descriptors that are key for image retrieval. \
User Queries: {queries} \
Synthesized Query:"""
PROMPT = """
Prompt: You are an AI assistant tasked with processing a series of natural \
language image search queries from a user. Your job is to analyze these \
queries, identify the core visual elements they are seeking, and combine \
these elements into a single, coherent query. This synthesized query should \
be optimized for searching against a vectorstore with CLIP embeddings, \
which means it needs to be clear, focused on visual elements and descriptors \
that are key for image retrieval, and stripped of any irrelevant details. \
{queries} \
Synthesized Query:"""


@st.cache_resource
def get_vectorstore(uid, _pinecone_index, _embedder):
    embed = _embedder
    text_field = "id"
    vectorstore = Pinecone(_pinecone_index, embed, text_field, namespace=uid)
    return vectorstore


def init_langchain(uid, _pinecone_index):
    vectorstore = get_vectorstore(uid, _pinecone_index)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["queries"],
        template=PROMPT,
    )
    llm = OpenAI(temperature=0.1)
    llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT)

    retriever_from_llm_chain = RePhraseQueryRetriever(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        llm_chain=llm_chain,
        k=50,
    )
    return retriever_from_llm_chain


@st.cache_resource
def load_inference():
    return InferenceClient(model="Salesforce/blip-image-captioning-large")


if "showing_results" not in st.session_state:
    st.session_state["showing_results"] = False

if "search_journey" not in st.session_state:
    st.session_state["search_journey"] = []

if "image_results" not in st.session_state:
    st.session_state["image_results"] = []

embedder = load_embedder()
blip_inference = load_inference()
pinecone_index = get_pinecone_image_index()
vectorstore: Pinecone = get_vectorstore(uid, pinecone_index, embedder)
llm_agent = init_langchain(uid, pinecone_index)
row_size = 5
top_k = 50
top_k_fewshot = 10


@st.cache_data
def query_images(search_journey):
    queries_string = ", ".join(search_journey)
    docs = vectorstore.similarity_search(
        query=queries_string, k=top_k_fewshot, namespace=f"{uid}_fewshot"
    )
    fewshots_query = ""
    if docs:
        fewshots_query = "Examples:\n"
        for doc in docs:
            fewshots_query += f"{doc.metadata['learnings']}\n"
    fewshots_query += f"User Queries: {', '.join(search_journey)}\n"

    # if len(search_journey) > 1:
    docs = llm_agent.get_relevant_documents(fewshots_query)
    # else:
    #     docs = vectorstore.similarity_search(query=queries_string, k=top_k)
    results = []
    for doc in docs:
        id = doc.page_content
        image_url = media_items_df.loc[media_items_df["id"] == id, "baseUrl"].iloc[0]
        results.append((image_url, id))

    return results


def click_search_button(query):
    if query == "":
        st.warning("Please enter a query to search.")
        return
    st.session_state.search_journey.append(query)
    st.session_state.image_results = query_images(st.session_state.search_journey)
    st.session_state.showing_results = True
    st.session_state["text_input_query"] = ""


def clear_search_journey():
    st.session_state.image_results = []
    st.session_state.search_journey = []
    st.session_state["text_input_query"] = ""
    st.session_state["showing_results"] = False


@st.cache_data
def get_image_caption(image_url):
    try:
        return blip_inference.image_to_text(image=image_url)
    except Exception as e:
        st.error(
            "Google Photos Base URL expired please go to Upsert Images and reupload images."
        )
        print(f"Error getting image caption: {e}")
        st.stop()


def learn_from_target_image(image_url_and_id):
    with st.spinner("Learning"):
        id = image_url_and_id[1]
        image_url = image_url_and_id[0]
        blip_caption = get_image_caption(image_url)
        queries_string = ", ".join(st.session_state.search_journey)
        few_shot_example = (
            f"User Queries: {queries_string} \nSynthesized Query: {blip_caption}"
        )
        st.info(f"Learned: {queries_string} -> {blip_caption}")
        print(f"INFO:3_Image_Search.py: learned: {queries_string} : {blip_caption}")

        caption_embedding = embedder.embed_query(queries_string)
        pinecone_index.upsert(
            vectors=[
                (
                    id,
                    caption_embedding,
                    {"id": image_url_and_id[1], "learnings": few_shot_example},
                )
            ],
            namespace=f"{uid}_fewshot",
            async_req=True,
        )
    clear_search_journey()


st.title("Storylines Search")
query = st.text_input("Search till you find it!", key="text_input_query")
st.button("Search", on_click=click_search_button, args=[query])

if st.session_state.showing_results:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.header("Most Relevant")
        image_results = st.session_state.image_results
        images = [x[0] for x in image_results]
        selection_id = image_select(
            label="Select your target image",
            images=images,
            return_value="index",
        )
        selection = image_results[selection_id]
    with col1:
        st.header("Target Image")
        st.image(selection[0])
        st.button("Accept", on_click=learn_from_target_image, args=[selection])
    with col3:
        st.header("Journey")
        search_journey_str = " <li>".join(st.session_state.search_journey)
        st.write(f"<ol><li>{search_journey_str}</ol>", unsafe_allow_html=True)
        st.button("Clear Search Journey", on_click=clear_search_journey)
else:
    st.header("Gallery")
    _col1, _col2 = st.columns([1, 2])
    with _col1:
        num_pages = int(len(media_items_df) / (row_size * (row_size + 1))) + 1
        page_number = st.number_input(
            label="Page Number", min_value=1, max_value=num_pages, value=1, step=1
        )
    with _col2:
        st.caption(
            "If images aren't rendering, please go to Upsert Images and reupload images."
        )
    grid = st.columns(row_size)
    col = 0
    batch_size = row_size * row_size
    start = (page_number - 1) * batch_size
    end = start + batch_size
    batch = media_items_df["baseUrl"].values[start:end]
    for i, image in enumerate(batch):
        with grid[col]:
            st.image(image)
        col = (col + 1) % row_size
