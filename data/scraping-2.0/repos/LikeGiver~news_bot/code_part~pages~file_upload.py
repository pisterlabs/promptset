import streamlit as st
import anthropic
from vector_store import get_vector_store, get_embed_model
from text_process import split_text_file

from pathlib import Path
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index.node_parser.text import SentenceSplitter
from llama_index.schema import TextNode

FILE_CACHE_PATH = '/home/ubuntu/data/tyk_code/news_bot/data'

vector_store = get_vector_store()

embed_model = get_embed_model()


uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"))

uploaded_text = st.text_area("Enter text", height=200)

metadata = {}

metadata["keys"] = st.text_area("Enter metadata", height=40)

if st.button("Submit") and len(uploaded_text) > 0 :
    st.write("uploading the text to vector store...")

    # save the text to local
    try:
        with open(FILE_CACHE_PATH + 'cache.txt', "w") as f:
            f.write(uploaded_text)
    except:
        st.write("Error while saving text.")

    with open(FILE_CACHE_PATH + 'cache.txt', "r") as f:
        text_original = f.read()

    text_chunks = split_text_file(text_original, max_chars_per_file=200, separator="\n")

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        print("chunk: ", idx)
        print(text_chunk)

        node = TextNode(
            text= metadata["keys"] + '\n\n' + str(text_chunk),
        )
        node.metadata = metadata if len(metadata) > 0 else "DEFAULT METADATA"
        nodes.append(node)
        
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
    
    vector_store.add(nodes)
    st.write("Done!")

# deal with the pdf file
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":

        #save uploaded_file to local
        try:
            with open(FILE_CACHE_PATH + uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except:
            st.write("Error while saving file.")

        loader = PyMuPDFReader()
        documents = loader.load(file_path=FILE_CACHE_PATH+ uploaded_file.name)
        
        text_parser = SentenceSplitter(
            chunk_size=200,
            # separator=" ",
        )

        text_chunks = []
        # maintain relationship with source doc index, to help inject doc metadata in (3)
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))

        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(
                text=text_chunk,
            )
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)
            
        for node in nodes:
            node_embedding = embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding
        
        vector_store.add(nodes)
        st.write("Done!")

    if uploaded_file.type == "text/markdown":

        #save uploaded_file to local
        try:
            with open('/home/likegiver/Desktop/codes/2023_11/nlp_final/our_work/data/'+ uploaded_file.name, "w") as f:
                f.write(uploaded_file.read())
        except:
            st.write("Error while saving file, maybe because there already exists a file with the same name.")

        with open('/home/likegiver/Desktop/codes/2023_11/nlp_final/our_work/data/'+ uploaded_file.name, "r") as f:
            documents = f.read()
        
        text_parser = SentenceSplitter(
            chunk_size=200,
            # separator=" ",
        )

        text_chunks = []
        # maintain relationship with source doc index, to help inject doc metadata in (3)
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = text_parser.split_text(doc)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))

        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(
                text=text_chunk,
            )
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
