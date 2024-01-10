from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth
import json
import pickle


def  get_wiki_space_child_pages(space_id):
    ids = []
    url = f"https://missionlane.atlassian.net/wiki/rest/api/content/{space_id}/descendant/page?limit=100"

    auth = HTTPBasicAuth("email goes here", "key goes here")

    headers = {
    "Accept": "application/json"
    }

    response = requests.request(
    "GET",
    url,
    headers=headers,
    auth=auth
    )

    data = json.loads(response.text)

    for id in data["results"]:
        ids.append(id["id"])
    return ids

def get_wiki_data(doc_id):
    url = f"https://missionlane.atlassian.net/wiki/rest/api/content/{doc_id}?expand=body.view"

    auth = HTTPBasicAuth("email goes here", "key goes here")

    headers = {
    "Accept": "application/json"
    }

    response = requests.request(
    "GET",
    url,
    headers=headers,
    auth=auth
    )

    data = json.loads(response.text)

    soup = BeautifulSoup(data["body"]["view"]["value"],features="html.parser")

    return Document(
        page_content=soup.get_text(),
        metadata={"source": data["_links"]["base"]+data["_links"]["tinyui"]},
    )


source_chunks = []

ids = get_wiki_space_child_pages("1781301249")

print("all ids are:", ids)

splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
# print(sources[0])

# loop through ids and call get_wiki_data
for id in ids:
    print("gettting data for id: ", id)
    doc = get_wiki_data(id)
    if doc.page_content is None or doc.page_content == "true" or doc.page_content == '':
        print("not including doc id: ", id)
    else:
        print("chunking doc id: ", id)
        for chunk in splitter.split_text(doc.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=doc.metadata))

with open("search_index.pickle", "wb") as f:
    pickle.dump(FAISS.from_documents(source_chunks, OpenAIEmbeddings()), f)