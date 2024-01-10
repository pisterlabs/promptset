import argparse
import json

import torch
from dotenv import load_dotenv
from langchain import LlamaCpp, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from prefect import task, flow

from repository.user_conversation import UserConversation
from util.file_util import get_file_path_by_key
from util.path_util import embeddingdb_path, bucket
from workflow import model_util

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

max_ctx_size = 2048 * 4


@task(name="create lager language model")
def create_lager_language_model() -> LlamaCpp:
    # return model_util.create_llama_model()
    return model_util.create_llama_model()


@task(name="create retriever")
def create_retriever(conversation_key: str) -> VectorStoreRetriever:
    persist_directory = get_file_path_by_key(bucket=bucket(), file_key=f'{embeddingdb_path()}/{conversation_key}')
    retriever = Chroma(
        persist_directory=persist_directory,
        embedding_function=HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={"device": device_type},
        ),
    ).as_retriever()
    return retriever


@task(name="create prompt template")
def create_prompt_template() -> PromptTemplate:
    # create prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    return PromptTemplate(input_variables=["history", "context", "question"], template=template)


@task(name="create retriever question answer")
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


@task(name="query document")
def query_document(retrieval_qa: RetrievalQA, query: str):
    return retrieval_qa(query)


@task(name="update answer of question in conversation")
def update_answer(conversation_key: str, user_id: str, conversation_id: str, question_time: int,
                  result: dict):
    dict_result = {
        "result": result.get("result"),
        "source_documents": list(map(lambda x: json.loads(x.json()), result.get("source_documents")))
    }
    UserConversation(user_id=user_id, conversation_id=conversation_id, question_time=question_time,
                     conversation_key=conversation_key).add_answer(answer=dict_result)


@flow(name="conversation asking flow")
def conversation_asking_flow(conversation_key: str, user_id: str, conversation_id: str, question_time: int,
                             question_text: str):
    llm = create_lager_language_model.submit()
    retriever = create_retriever.submit(conversation_key=conversation_key)
    prompt = create_prompt_template.submit()
    retrieval_qa = create_retriever_qa.submit(llm=llm, retriever=retriever, prompt=prompt)
    result = query_document.submit(retrieval_qa=retrieval_qa, query=question_text)
    update_answer.submit(user_id=user_id, conversation_id=conversation_id,
                         question_time=question_time, result=result, conversation_key=conversation_key)


main_args = {
    "conversation_key": "conversation_key",
    "user_id": "user_id",
    "conversation_id": "conversation_id",
    "question_time": "question_time",
    "question_text": "question_text"
}


def main(kwargs: argparse.Namespace):
    conversation_key = kwargs.conversation_key
    user_id = kwargs.user_id
    conversation_id = kwargs.conversation_id
    question_time = kwargs.question_time
    question_text = kwargs.question_text

    conversation_asking_flow(conversation_key=conversation_key, user_id=user_id, conversation_id=conversation_id,
                             question_text=question_text, question_time=question_time)


if __name__ == '__main__':
    load_dotenv()

    parser = argparse.ArgumentParser()
    for key, value in main_args.items():
        parser.add_argument(f"--{key}", help=value)

    main(parser.parse_args())
