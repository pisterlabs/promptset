import tiktoken
import streamlit as st
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)


def create_retriever(embeddings, splits):
    try:
        vectorstore = FAISS.from_documents(splits, embeddings)
    except (IndexError, ValueError) as e:
        st.error(f"Error creating vectorstore: {e}")
        return
    retriever = vectorstore.as_retriever(search_type="mmr")
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10

    return retriever


def chunking_string(all_tokens, chunking_size, overlap_size):
    tmp_idx = 0
    sliced_lists = []

    tmp_list = []
    while tmp_idx <= len(all_tokens) - 1:
        if tmp_idx == 0:
            tmp_list = all_tokens[:chunking_size]
            tmp_idx = chunking_size - overlap_size
        else:
            tmp_list = all_tokens[tmp_idx : tmp_idx + chunking_size]
            tmp_idx += chunking_size
        sliced_lists.append(tmp_list)

    return sliced_lists


@st.cache_data(show_spinner=False)
def dictionary_to_docs(
    github_info_dict,
    structure_content,
    user_content,
    chunking_size,
    overlap_size,
    model_name,
):
    ret_docs = []
    language_splitter = {
        ".py": Language.PYTHON, ".js": Language.JS,
        ".cpp": Language.CPP, ".go": Language.GO,
        ".java": Language.JAVA, ".php": Language.PHP,
        ".proto": Language.PROTO, ".rst": Language.RUST,
        ".scala": Language.SCALA, ".swift": Language.SWIFT,
        ".md": Language.MARKDOWN, ".tex": Language.LATEX,
        ".html": Language.HTML, ".sol": Language.SOL,
    }

    for file_name, file_content in github_info_dict.items():
        tmp_docs = []
        file_extension = "." + file_name.split(".")[-1]

        if file_extension in language_splitter:
            file_splitter = RecursiveCharacterTextSplitter.from_language(
                language=language_splitter[file_extension],
                chunk_size=chunking_size,
                chunk_overlap=overlap_size,
            )
            tmp_docs = file_splitter.create_documents([file_content])
        else:
            encoding = tiktoken.encoding_for_model(model_name)
            total_encoded = encoding.encode(file_content)
            chunked_string = chunking_string(total_encoded, chunking_size, overlap_size)

            for tmp_encoded in chunked_string:
                tmp_doc = Document(
                    page_content=encoding.decode(tmp_encoded),
                    metadata={"source": file_name},
                )
                tmp_docs.append(tmp_doc)

        for tmp_doc in tmp_docs:
            tmp_doc.metadata["source"] = file_name
        ret_docs.extend(tmp_docs)

    structure_doc = Document(
        page_content=structure_content, metadata={"source": "github_folder_structure"}
    )
    ret_docs.append(structure_doc)

    user_doc = Document(
        page_content=user_content, metadata={"source": "github_user_information"}
    )
    ret_docs.append(user_doc)

    return ret_docs
