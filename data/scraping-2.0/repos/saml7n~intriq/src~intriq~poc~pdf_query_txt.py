from reportlab.platypus import SimpleDocTemplate, Table
import fitz
import ssl
import os
from unstructured.partition.auto import partition
from unstructured.partition.xlsx import partition_xlsx
from unstructured.staging.base import convert_to_dict
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import streamlit as st
import pandas as pd
from loguru import logger
from dotenv import main
from chromadb.utils import embedding_functions
from langchain.callbacks import StreamlitCallbackHandler
from langchain.docstore.document import Document
import tempfile
import nltk

######################
import torch
from transformers import pipeline
from llama_index import LLMPredictor, KnowledgeGraphIndex, Prompt, ServiceContext, SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import SimpleGraphStore
from llama_index.text_splitter import SentenceSplitter

main.load_dotenv()
_PARTITION_STRATEGY = 'hi_res'
_PARTITION_MODEL_NAME = 'yolox'
_OPEN_AI_MODEL_NAME = 'gpt-3.5-turbo-0613'

# Download necessary NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

st.set_page_config(
    page_title="intriq data chatbot", page_icon="🦜")
st.title("🔥🤖🔥 intriq data chatbot 🔥🤖🔥")


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False

# def _highlight_table(table):
#     tabs = page.find_tables()  # detect the tables
#     for i,tab in enumerate(tabs):  # iterate over all tables
#         for cell in tab.header.cells:
#             page.draw_rect(cell,color=fitz.pdfcolor["red"],width=0.3)
#         page.draw_rect(tab.bbox,color=fitz.pdfcolor["green"])
#         print(f"Table {i} column names: {tab.header.names}, external: {tab.header.external}")

#     show_image(page, f"Table & Header BBoxes")


def _extract_tables(pdf_file):
    docs = []
    for page in pdf_file:
        tabls = page.find_tables()
        for i, tabl in enumerate(tabls):
            logger.info(f'Table {i} column names: {tabl.header.names}')
            tabl_df = tabl.to_pandas()
            tabl_df = tabl_df.replace('\n', ' ', regex=True)
            str_to_embed = ''
            for index, row in tabl_df.iterrows():
                row_str = ''
                for col in tabl_df.columns:
                    row_str += f'{col}: {row[col]}, '
                formatted = row_str[:-2]
                str_to_embed += formatted + "\n"
            docs.append(
                Document(
                    page_content=str_to_embed[:-2],
                    metadata={'headers': ', '.join(tabl.header.names)}
                )
            )
    return docs


# drop files here
uploaded_files = st.file_uploader(
    "Drop all your shit here 💩",
    accept_multiple_files=True,
    help="Various File formats are supported",
    on_change=clear_submit,
)

if not uploaded_files:
    st.warning(
        "What am I a mind reader? Upload some data files if you want to chat with me."
    )

# pipe = pipeline(
#     "text-generation",
#     model="HuggingFaceH4/zephyr-7b-beta",
#     # torch_dtype='auto',
#     device_map="auto"
# )

llm = ChatOpenAI(
    model_name=_OPEN_AI_MODEL_NAME,
    temperature=0,
    streaming=True
)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=OpenAIEmbeddings(),
    text_splitter=SentenceSplitter(chunk_size=1024, chunk_overlap=20))
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)


if uploaded_files:
    loaders = []
    for uploaded_file in uploaded_files:
        try:
            ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
        except:
            ext = uploaded_file.split(".")[-1]
        if ext.lower() not in ['pdf']:
            logger.warning('not a PDF. Skipping')
            continue
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        pdf_file = SimpleDirectoryReader(
            'data').load_data()
        # pdf_file = fitz.open(tmp_file_path)
        # pdf_txt = chr(12).join([page.get_text() for page in pdf_file])
        # docs = _extract_tables(pdf_file)
        doc = {
            'allowed_nodes': [],
            'allowed_rels': []
        }

        # Replace with your list of allowed nodes
        allowed_nodes = doc.get('allowed_nodes', ['table', 'system'])
        # Replace with your list of allowed relationships
        allowed_rels = doc.get('allowed_rels', ['causes', 'influences'])

        template = (f"""
    You are a top-tier algorithm designed for extracting information in structured formats to build an ontology.
    - **Classes** represent categories or types of entities. They're akin to Wikipedia categories.
    - **Instances** are the individual entities belonging to these classes.
    - **Properties** define attributes and relationships between instances and classes.
    
    Here are the guidelines you should follow:

    ## 1. Identifying and Categorizing Entities
    - **Consistency**: Use consistent, basic types for class labels.
    - **Instance IDs**: Use names or human-readable identifiers found in the text as instance IDs. Avoid using integers.
    - **Allowed Class Labels:** {", ".join(allowed_nodes)}
    - **Allowed Property Types:** {", ".join(allowed_rels)}

    ## 2. Defining Hierarchical Relationships
    - Identify superclass-subclass relationships between classes. For example, if you identify "bird" and "sparrow", establish that "sparrow" is a subclass of "bird".

    ## 3. Handling Numerical Data and Dates
    - Numerical data should be incorporated as properties of the respective instances or classes.
    - Do not create separate classes for dates or numerical values. Always attach them as properties.

    ## 4. Coreference Resolution
    - Maintain consistency when extracting entities. If an entity is referred to by different names or pronouns, always use the most complete identifier.

    ## 5. Inferring New Knowledge
    - Use the existing information in the ontology to infer new knowledge. For example, if "sparrows" are a type of "bird", and "birds" can "fly", infer that "sparrows" can "fly".

    ## 6. Strict Compliance
    - Adhere to these rules strictly. Non-compliance will result in termination.
    """)
        qa_template = Prompt(template)
        index = KnowledgeGraphIndex.from_documents(
            pdf_file,
            query_keyword_extract_template=qa_template,
            storage_context=storage_context,
            max_triplets_per_chunk=20,
            service_context=service_context,
            include_embeddings=True,
        )
        query_engine = index.as_query_engine(
            include_text=False,
            response_mode="tree_summarize"
        )
        # text_splitter = CharacterTextSplitter(
        #     chunk_size=1000, chunk_overlap=200
        # )
        # chunks = text_splitter.split_documents(docs)
        # embeddings = OpenAIEmbeddings()
        # db = Chroma.from_documents(docs, embeddings)


if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Whaddup? Ask me something about something. Or don't, that's fine.."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # llm = ChatOpenAI(
    #     model_name=_OPEN_AI_MODEL_NAME,
    #     temperature=0,
    #     streaming=True
    # )

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=db.as_retriever(search_kwargs={"k": 5}),
    #     return_source_documents=True
    # )

    with st.chat_message("assistant"):
        # st_cb = StreamlitCallbackHandler(
        #     st.container(), expand_new_thoughts=False)
        # docs = db.similarity_search(prompt)
        response = query_engine.query(prompt)
        # response = qa_chain(
        #     {'query': prompt},
        #     callbacks=[st_cb]
        # )
        # response = index.query_with_sources(prompt)
        # logger.info(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.write(response)
