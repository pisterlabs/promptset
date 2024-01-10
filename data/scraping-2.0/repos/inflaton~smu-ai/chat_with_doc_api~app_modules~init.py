"""Main entrypoint for the app."""
import os
from timeit import default_timer as timer
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS

from app_modules.llm_loader import LLMLoader
from app_modules.llm_qa_chain import QAChain
from app_modules.utils import get_device_types, init_settings

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

# Constants
init_settings()

llm_loader = None
qa_chain = None


def load_vectorstor(using_faiss, index_path, embeddings):
    start = timer()

    print(f"Load index from {index_path} with {'FAISS' if using_faiss else 'Chroma'}")

    if not os.path.isdir(index_path):
        raise ValueError(f"{index_path} does not exist!")
    elif using_faiss:
        vectorstore = FAISS.load_local(index_path, embeddings)
    else:
        vectorstore = Chroma(
            embedding_function=embeddings, persist_directory=index_path
        )

    end = timer()

    print(f"Completed in {end - start:.3f}s")
    return vectorstore


def app_init(initQAChain: bool = True):
    global llm_loader
    global qa_chain
    if llm_loader == None:
        # https://github.com/huggingface/transformers/issues/17611
        os.environ["CURL_CA_BUNDLE"] = ""

        llm_model_type = os.environ.get("LLM_MODEL_TYPE")
        n_threds = int(os.environ.get("NUMBER_OF_CPU_CORES") or "4")

        hf_embeddings_device_type, hf_pipeline_device_type = get_device_types()
        print(f"hf_embeddings_device_type: {hf_embeddings_device_type}")
        print(f"hf_pipeline_device_type: {hf_pipeline_device_type}")

        if initQAChain:
            hf_embeddings_model_name = (
                os.environ.get("HF_EMBEDDINGS_MODEL_NAME") or "hkunlp/instructor-xl"
            )

            index_path = os.environ.get("FAISS_INDEX_PATH") or os.environ.get(
                "CHROMADB_INDEX_PATH"
            )
            using_faiss = os.environ.get("FAISS_INDEX_PATH") is not None

            start = timer()
            embeddings = HuggingFaceInstructEmbeddings(
                model_name=hf_embeddings_model_name,
                model_kwargs={"device": hf_embeddings_device_type},
            )
            end = timer()

            print(f"Completed in {end - start:.3f}s")

            vectorstore = load_vectorstor(using_faiss, index_path, embeddings)

            doc_id_to_vectorstore_mapping = {}
            rootdir = index_path
            for file in os.listdir(rootdir):
                d = os.path.join(rootdir, file)
                if os.path.isdir(d):
                    v = load_vectorstor(using_faiss, d, embeddings)
                    doc_id_to_vectorstore_mapping[file] = v

            # print(doc_id_to_vectorstore_mapping)

        start = timer()
        llm_loader = LLMLoader(llm_model_type)
        llm_loader.init(
            n_threds=n_threds, hf_pipeline_device_type=hf_pipeline_device_type
        )
        qa_chain = (
            QAChain(vectorstore, llm_loader, doc_id_to_vectorstore_mapping)
            if initQAChain
            else None
        )
        end = timer()
        print(f"Completed in {end - start:.3f}s")

    return llm_loader, qa_chain
