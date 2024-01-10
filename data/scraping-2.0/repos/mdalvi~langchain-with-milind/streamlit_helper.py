# noinspection PyUnresolvedReferences
import json
import os
import shutil
from typing import Tuple

import faiss
import joblib
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA

# noinspection PyUnresolvedReferences
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from parsers.pydantic import PersonalIntel
from third_parties.linkedin import get_linkedin_profile, get_saved_linkedin_profile
from tools.regex import get_linkedin_username
from tools.requests import get_image_from_url

GENDER_PREFIX = [
    "Mr.",
    "Dr.",
    "Prof.",
    "Sir",
    "Ms.",
    "Mrs.",
    "Miss",
    "Prof.",
    "Madam",
    "Ma'am",
]


# Define the metadata extraction function.
def _metadata_func(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    metadata["start"] = record.get("start")
    metadata["end"] = record.get("end")
    if "source" in metadata:
        start = record.get("start")
        video_id = record.get("video_id")
        video_url = f"storage/videos/{video_id}.mp4"
        metadata["source"] = json.dumps({"start": start, "video_url": video_url})
    return metadata


def run_retrival_qna_chain_for_video(video_id: str) -> BaseRetrievalQA:
    contexts_file_path = f"storage/videos/{video_id}_contexts.json"
    transcript = joblib.load(f"storage/videos/{video_id}_transcript.joblib")
    contexts = {
        "contexts": [
            {
                "start": segment["start"],
                "text": segment["text"],
                "video_id": video_id,
            }
            for segment in transcript["segments"]
        ]
    }
    print(
        f"There are total #{contexts['contexts'].__len__()} contexts / whisper segments in this file"
    )

    # nb_contexts = contexts["contexts"].__len__()
    # divisor = nb_contexts // 10
    # reduced_contexts, context_text = [], ""
    # for index, dict_ in enumerate(contexts["contexts"]):
    #     if index % divisor == 0:
    #         context_text += dict_["text"] + " "
    #         reduced_contexts.append({"text": context_text, "start": dict_["start"]})
    #         context_text = ""
    #     else:
    #         context_text += dict_["text"] + " "
    #
    # contexts["contexts"] = reduced_contexts
    # print(f"There are total #{contexts['contexts'].__len__()} reduced contexts")

    with open(contexts_file_path, "w") as f:
        f.write(json.dumps(contexts))
    doc_loader = JSONLoader(
        contexts_file_path,
        jq_schema=".contexts[]",
        content_key="text",
        metadata_func=_metadata_func,
    )
    contexts = doc_loader.load()
    print(f"Number of contexts created by splitter: # {len(contexts)}")
    embeddings = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vector_store = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id=dict(),
    )
    vector_store.add_documents(contexts)
    prompt_template = """Use the following pieces of context to answer the question at the end. If the answer is not available in the provided context, just say that you don't know, don't try to make up an answer. You may however respond to greetings.
    
            Aiman Ezzat, CEO, shares the highlights from our strong half year performance, and explains how thanks to a strong strategic positioning, we continue to gain market share as we accompany our clients in their transition towards a digital and sustainable economy. He also commented on the Group's plan of investing €2 billion over the next 3 years to strengthen its leadership in Artificial Intelligence.
            {context}

            Question: {question}
            Answer:"""
    custom_template = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": custom_template}
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
    )
    return chain


def run_retrival_qna_chain_for_pdf_document(file_path: str) -> BaseRetrievalQA:
    """
    Demonstrates guard rails using prompt engineering
    :return:
    """
    doc_loader = PyPDFLoader(file_path)
    raw_documents = doc_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    contexts = splitter.split_documents(raw_documents)
    print(f"Number of contexts created by splitter: # {len(contexts)}")
    embeddings = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vector_store = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id=dict(),
    )
    vector_store.add_documents(contexts)
    prompt_template = """Use the following pieces of context to answer the question at the end. If the answer is not available in the provided context, just say that you don't know, don't try to make up an answer. You may however respond to greetings.

            {context}

            Question: {question}
            Answer:"""
    custom_template = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": custom_template}
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 15, "fetch_k": 50}
        ),  # search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50}
        chain_type_kwargs=chain_type_kwargs,
    )
    return chain


def run_chain_for_linkedin(
    linkedin_url: str, prefix: str
) -> Tuple[PersonalIntel, dict]:
    """
    Step 1: Fetches public data from LinkedIn about the person using the url
    Step 2: Generates summary and two interesting facts about the person using LLM and above information
    :return: None
    """
    user_name = get_linkedin_username(profile_url=linkedin_url)
    file_path = f"storage/linkedin/{user_name}.json"
    if os.path.exists(file_path):
        print("Using saved profile to process request.")
        profile_data = get_saved_linkedin_profile(f"storage/linkedin/{user_name}.json")
    else:
        print("Invoking nubela.co for profile details.")
        profile = get_linkedin_profile(profile_url=linkedin_url)
        with open(f"storage/linkedin/{user_name}.json", "w") as f:
            f.write(json.dumps(profile))
        profile_data = get_saved_linkedin_profile(f"storage/linkedin/{user_name}.json")

    image_path = f"storage/linkedin/images/{user_name}.jpg"
    if not os.path.exists(image_path):
        if not get_image_from_url(
            profile_data["profile_pic_url"], f"storage/linkedin/images/{user_name}.jpg"
        ):
            shutil.copy(
                f"storage/linkedin/images/no_image.jpg",
                image_path,
            )
    profile_data.pop("profile_pic_url", None)

    out_parser = PydanticOutputParser(pydantic_object=PersonalIntel)
    custom_template = """
        Given the LinkedIn information {profile_data} about a person, create
        1. A short summary
        2. Two interesting facts about the person, use gender prefix as {prefix}
        3. A topic that may interest them
        4. Two creative Ice-breakers to open a conversation with them
        \n{format_instructions}
        """

    prompt = PromptTemplate(
        input_variables=["profile_data", "prefix"],
        template=custom_template,
        partial_variables={"format_instructions": out_parser.get_format_instructions()},
    )
    azure_credentials = {
        "temperature": 0,
        "deployment_name": os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        "openai_api_type": os.environ["AZURE_OPENAI_API_TYPE"],
        "openai_api_base": os.environ["AZURE_OPENAI_API_BASE"],
        "openai_api_version": os.environ["AZURE_OPENAI_API_VERSION"],
        "openai_api_key": os.environ["AZURE_OPENAI_API_KEY"],
    }
    llm = AzureChatOpenAI(**azure_credentials)
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)
    return (
        out_parser.parse(chain.run(profile_data=profile_data, prefix=prefix)),
        {
            "image_path": image_path,
            "full_name": profile_data["full_name"],
            "headline": profile_data["headline"],
        },
    )
