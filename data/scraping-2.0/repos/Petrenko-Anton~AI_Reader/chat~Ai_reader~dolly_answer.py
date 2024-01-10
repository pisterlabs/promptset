import torch
import transformers
from langchain.vectorstores import FAISS
import pickle
from pdf.models import UploadedFile


def gen_text(context: str, question: str) -> str:
    generate_text = transformers.pipeline(
        model="databricks/dolly-v2-3b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        # offload_folder="",
        return_full_text=True,
    )

    res = generate_text(f"{context}. {question}")
    return res[0]["generated_text"]


def answer(filename: UploadedFile, question: str) -> str:
    vector_store = pickle.load(filename.vector_db)

    context = vector_store.similarity_search(query=question, k=3)
    return gen_text(context, question)
