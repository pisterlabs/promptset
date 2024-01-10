import torch
import git
import os
import textwrap

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceInstructEmbeddings

config = {"max_new_tokens": 1024, "temperature": 0.1, "context_length": 2048}
llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    config=config,
    n_ctx=4096,
)


def processGitLink(git_link) -> None:
    last_name = git_link.split("/")[-1]
    clone_path = last_name.split(".")[0]
    return clone_path


def clone_repo(git_link, clone_path):
    if not os.path.exists(clone_path):
        git.Repo.clone_from(git_link, clone_path)
        return


def extract_all_files(clone_path, allowed_extensions):
    docs = []

    root_dir = clone_path
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            file_extension = os.path.splitext(file)[1]
            if file_extension in allowed_extensions:
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                    docs.extend(loader.load())
                except Exception as e:
                    pass
    return docs


def chunk_files(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    return texts


def create_vectordb(texts):
    instructor_embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda:0"}
    )
    embedding = instructor_embeddings

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
    )

    return vectordb


def retriever_pipeline(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa_chain


def wrap_text_preserve_newlines(text, width=110):
    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = "\n".join(wrapped_lines)
    return wrapped_text


def process_llm_response(llm_response):
    wrapped_text = wrap_text_preserve_newlines(llm_response["result"])
    print(wrapped_text)
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])
    return wrapped_text
