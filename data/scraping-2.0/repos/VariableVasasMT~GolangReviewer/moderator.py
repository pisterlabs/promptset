
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import ConfluenceLoader

load_dotenv(override=True)

### ENV VARIABLES ###
OPEN_AI_TOKEN=os.getenv("OPENAI_API_KEY")
ACTIVELOOP_TOKEN=os.getenv("ACTIVELOOP_TOKEN")
ATLASSIAN_API_TOKEN=os.getenv("ATLASSIAN_API_TOKEN")
ATLASSIAN_INSTANCE_URL=os.getenv("ATLASSIAN_INSTANCE_URL")
ATLASSIAN_USERNAME=os.getenv("ATLASSIAN_USERNAME")

CONFLUENCE_SPACES=[
    os.getenv("DEVOPS_CONFLUENCE_SPACE"), 
    os.getenv("CODE_INSPECTORS_CONFLUENCE_SPACE")
]
DEEPLAKE_DATASET_PATH=os.getenv("DEEPLAKE_DATASET_PATH")
CLONED_REPO_PATH = os.getenv('CLONED_REPO_PATH')


### CONFLUENCE DOCS ###



CONFLUENCE = 'confluence'
GITLAB = 'gitlab'

print(ATLASSIAN_API_TOKEN)

def index_confluence_docs():
    docs = []
    for space in CONFLUENCE_SPACES:
        confluence = ConfluenceLoader(
            url=ATLASSIAN_INSTANCE_URL,
            # username=ATLASSIAN_USERNAME,
            token=ATLASSIAN_API_TOKEN
        )
        docs.extend(confluence.load(space_key=space, include_attachments=True, limit=50))
    return docs


### GIT REPOS ###
def index_repos():
    docs = []
    for dirpath, dirnames, filenames in os.walk(CLONED_REPO_PATH):  
        print(f"Processing {dirpath} || {dirnames} || {filenames}")
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass
    return docs

### DEEPLAKE INGESTION ###
def ingest_into_deeplake(docs, source=GITLAB):
    embeddings = OpenAIEmbeddings(disallowed_special=())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    dataset_path = DEEPLAKE_DATASET_PATH
    db = DeepLake.from_documents(texts, dataset_path=dataset_path, embedding=embeddings, source=source )
    return db

def main():
    # docs_repos = index_repos()
    # ingest_into_deeplake(docs_repos, source=GITLAB)

    docs_confluence = index_confluence_docs()
    ingest_into_deeplake(docs_confluence, source=CONFLUENCE)

if __name__ == "__main__":
    main()


