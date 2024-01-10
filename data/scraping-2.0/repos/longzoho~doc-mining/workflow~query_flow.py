import json

from huggingface_hub import hf_hub_download
from langchain import LlamaCpp, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from prefect import task, flow
import torch
from util.file_util import get_file_path_by_key
from util.path_util import embeddingdb_path, bucket

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

max_ctx_size = 2048*4
model_id = "TheBloke/Llama-2-7B-Chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"

# download model
hf_hub_download(repo_id=model_id, filename=model_basename)


@task
def create_lager_language_model() -> LlamaCpp:
    kwargs = {
        "model_path": hf_hub_download(repo_id=model_id, filename=model_basename),
        "n_ctx": max_ctx_size,
        "max_tokens": max_ctx_size,
        "n_gpu_layers": 1000 if device_type.lower() in ["mps", "cuda"] else None,
        "n_batch": max_ctx_size if device_type.lower() == "cuda" else None,
        "verbose": False
    }

    # create lager language model
    return LlamaCpp(**kwargs)


@task
def create_retriever(profile_id: str) -> Chroma:
    persist_directory = get_file_path_by_key(bucket=bucket(), file_key=f'{embeddingdb_path()}/{profile_id}')
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={"device": device_type},
        ),
    ).as_retriever()


@task
def create_prompt_template() -> PromptTemplate:
    # create prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    return PromptTemplate(input_variables=["history", "context", "question"], template=template)


@task
def create_retriever_qa(llm: LlamaCpp, retriever: Chroma, prompt: PromptTemplate) -> RetrievalQA:
    # create memory
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    # create retriever question answer
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )


@task
def query_document(retrieval_qa: RetrievalQA, query: str):
    return retrieval_qa(query)


@flow
def query_flow(profile_id: str, query: str):
    llm = create_lager_language_model.submit()
    retriever = create_retriever.submit(profile_id=profile_id)
    prompt = create_prompt_template.submit()
    retrieval_qa = create_retriever_qa.submit(llm=llm, retriever=retriever, prompt=prompt)
    result = query_document(retrieval_qa=retrieval_qa, query=query)
    dict_result = {
        "result": result.get("result"),
        "source_documents": list(map(lambda x: json.loads(x.json()), result.get("source_documents")))
    }
    return dict_result
