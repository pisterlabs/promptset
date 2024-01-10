import tiktoken
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
import streamlit as st

def chunking_string(all_tokens , chunking_size, overlap_size):
    tmp_idx = 0
    sliced_lists = []

    tmp_list = []
    while tmp_idx <= len(all_tokens)-1:
        if tmp_idx == 0:
            tmp_list = all_tokens[:chunking_size]
            tmp_idx = chunking_size-overlap_size
        else:
            tmp_list = all_tokens[tmp_idx:tmp_idx+chunking_size]
            tmp_idx += chunking_size
        sliced_lists.append(tmp_list)

    return sliced_lists

@st.cache_data()
def dictionary_to_docs(github_info_dict, structure_content, chunking_size, overlap_size, model_name):
    ret_docs = []
    for file_name, file_content in github_info_dict.items():
        tmp_docs = []
        if file_name.endswith(".py"):
            py_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=chunking_size, chunk_overlap=overlap_size
            )
            tmp_docs = py_splitter.create_documents([file_content])
        elif file_name.endswith(".js"):
            js_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS, chunk_size=chunking_size, chunk_overlap=overlap_size)
            tmp_docs = js_splitter.create_documents([file_content])
        else:
            encoding = tiktoken.encoding_for_model(model_name)
            total_encoded = encoding.encode(file_content)
            chunked_string = chunking_string(total_encoded, chunking_size, overlap_size)

            for tmp_encoded in chunked_string:
                tmp_doc = Document(
                        page_content=encoding.decode(tmp_encoded),
                        metadata={"source":file_name}
                    )
                tmp_docs.append(tmp_doc)

            for tmp_doc in tmp_docs:
                tmp_doc.metadata["source"] = file_name

    ret_docs.extend(tmp_docs)
    
    structure_doc = Document(
    page_content=structure_content,
    metadata={"source":"github_folder_structure"}
    )
    
    ret_docs.append(structure_doc)
    
    return ret_docs
