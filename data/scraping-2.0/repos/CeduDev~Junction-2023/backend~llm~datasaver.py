from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, SeleniumURLLoader, PlaywrightURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
import json

def datasaver(number):
    load_dotenv()

    key = os.environ["OPENAI_API_KEY"]
    faiss_folder = ""
    documents = None

    if(number == 1):
        raw_documents = TextLoader('./data/state_of_the_union.txt').load()
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        faiss_folder = "state_of_the_union"
    elif (number == 2):
        urls = [
            "https://www.infofinland.fi/en/information-about-finland/finnish-climate"
        ]

        raw_documents = SeleniumURLLoader(urls=urls).load()
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        faiss_folder = "finnish_climate"
    elif (number == 3):
        urls = [
            "https://vm.fi/en/frontpage"
        ]

        raw_documents = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"]).load()
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        faiss_folder = "vm"
    else:
        return 0

    embeddings = OpenAIEmbeddings(openai_api_key=key)
    # Create vectors
    vectors = FAISS.from_documents(documents, embeddings)
    # Persist the vectors locally on disk
    vectors.save_local("./faiss/{}".format(faiss_folder))

    return 1

def datasaver_url(dataset, name, url):

        load_dotenv()

        key = os.environ["OPENAI_API_KEY"]
        faiss_folder = dataset
        urls = [url]

         # fetch previous data from dataset

        f = open('./faiss/sources.json')
        sources = json.load(f)
        f.close()
        new_dataset = True
        for d in sources:
            if d["name"] == dataset:
                new_dataset = False
                for s in d["sources"]:
                    urls.append(s["source"])
                d["sources"].append(
                    {
                    "name": name,
                    "source": url,
                    "docType": "url"
                    }
                )   
                    
        if(new_dataset):
            sources.append(
                {
                    "name": dataset,
                     "id": str(len(sources)),
                    "sources": [
                    {
                        "name": name,
                        "source": url,
                        "docTypte": "url"
                    }]
                }
            )

        with open("./faiss/sources.json", "w") as outfile:
            json.dump(sources, outfile)


        raw_documents = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"]).load()
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        embeddings = OpenAIEmbeddings(openai_api_key=key)
        # Create vectors
        vectors = FAISS.from_documents(documents, embeddings)
        # Persist the vectors locally on disk
        vectors.save_local("./faiss/{}".format(faiss_folder))
        return