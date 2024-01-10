# -*- coding:utf-8 -*-
# Created by liwenw at 6/30/23

from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import os
from chromadb.config import Settings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from omegaconf import OmegaConf
import argparse
from templates import system_provider_template, human_provider_template, system_patient_template, human_patient_template

def create_parser():
    parser = argparse.ArgumentParser(description='demo how to use ai embeddings to question/answer.')
    parser.add_argument("-y", "--yaml", dest="yamlfile",
                        help="Yaml file for project", metavar="YAML")
    parser.add_argument("-r", "--role", dest="role",
                        help="role(patient/provider) for question/answering", metavar="ROLE")
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.yamlfile is None:
        parser.print_help()
        exit()

    yamlfile = args.yamlfile
    config = OmegaConf.load(yamlfile)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        openai_api_key = config.openai.api_key

    # Load environment variables
    load_dotenv()

    model = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=config.openai.chat_model_name,
        temperature=0.0,
        verbose=True
    )

    embeddings = OpenAIEmbeddings()

    collection_name = config.chromadb.collection_name
    persist_directory = config.chromadb.persist_directory
    chroma_db_impl = config.chromadb.chroma_db_impl

    vector_store = Chroma(collection_name=collection_name,
                          embedding_function=embeddings,
                          client_settings=Settings(
                              chroma_db_impl=chroma_db_impl,
                              persist_directory=persist_directory
                          ),
                          )

    chat_search_type = config.openai.chat_search_type
    chat_search_k = config.openai.chat_search_k
    retriever = vector_store.as_retriever(search_type=chat_search_type, search_kwargs={"k": chat_search_k})

    if args.role == "provider":
        system_template = system_provider_template
        human_template = human_provider_template
    elif args.role == "patient":
        system_template = system_patient_template
        human_template = human_patient_template
    else:
        print("role not supported")
        exit()

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        reduce_k_below_max_tokens=True,
        max_tokens_limit=8192,
        chain_type_kwargs=chain_type_kwargs,
        verbose=False,
    )

    while True:
        print()
        question = input("Question: ")

        if question == "exit":
            break

        # Get answer
        response = chain(question)
        answer = response["answer"]
        source = response["source_documents"]

        # Display answer
        print("\nSources:")
        for document in source:
            print(document)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
