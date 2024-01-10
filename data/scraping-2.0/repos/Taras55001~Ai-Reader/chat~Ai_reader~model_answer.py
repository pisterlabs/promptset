from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain

from transformers.utils import logging
import pickle
from pdf.models import UploadedFile

logging.set_verbosity_error


def gen_text(context: str, question: str) -> str:
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 455}
    )

    chain = load_qa_chain(llm, chain_type="stuff")
    res = chain.run(input_documents=context, question=question)
    return res


def answer(filename: UploadedFile, question: str) -> str:
    vector_store = pickle.load(filename.vector_db)

    context = vector_store.similarity_search(query=question, k=3)
    return gen_text(context, question)