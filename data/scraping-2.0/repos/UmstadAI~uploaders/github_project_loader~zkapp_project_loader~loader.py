import glob
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY")
import pinecone
import time
import re
import base64

from github import Github
from uuid import uuid4

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(), override=True)  # read local .env file

metadata_fields = {
    "project_name",
    "project_description",
    "file_name_with_folder",
    "comments",
}

token = os.getenv("GITHUB_ACCESS_TOKEN") or "GITHUB_ACCESS_TOKEN"

pinecone_api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or "YOUR_ENV"

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "zkappumstad"

# Delete/Comment if you want to upload MORE
""" if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

pinecone.create_index(
    name=index_name,
    metric='dotproduct',
    dimension=1536
) 

time.sleep(5) """


def project_loader(owner, project_name):
    g = Github(token)
    repo = g.get_repo(f"{owner}/{project_name}")

    base_dir = f"./projects/{project_name}"
    project_description = repo.description

    def export_project_description_from_readme(content):
        decoded_content = bytes(str(content), "utf-8").decode("unicode_escape")
        cleaned_content = re.sub(r"# ", "", decoded_content)
        cleaned_content = re.sub(r"#", "", cleaned_content)

        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )

        cleaned_content = re.sub(r"```.*?```", "", cleaned_content, flags=re.DOTALL)
        cleaned_content = emoji_pattern.sub(r"", cleaned_content)

        return cleaned_content[:850]

    if project_description is None:
        read_me = repo.get_readme()
        project_description = export_project_description_from_readme(
            base64.b64decode(read_me.content)
        )
        print("Project description from README.md", project_description)

    loader = GenericLoader.from_filesystem(
        base_dir, glob="**/*", suffixes=[".ts"], parser=LanguageParser(),
    )

    docs = loader.load()

    ts_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.TS, chunk_size=1200, chunk_overlap=200
    )

    docs = ts_splitter.split_documents(docs)

    model_name = "text-embedding-ada-002"

    def extract_comments_from_ts_code(ts_code):
        comment_pattern = r"(\/\/[^\n]*|\/\*[\s\S]*?\*\/)"
        comments = re.findall(comment_pattern, ts_code)
        comments_string = " ".join(
            comment.strip("/*").strip("*/").strip("//").strip() for comment in comments
        )

        return comments_string

    texts = []
    metadatas = []

    for doc in docs:
        metadata = {
            "Project Name": project_name,
            "Project Description": project_description,
            "File Name": doc.metadata["source"],
            "Project content": extract_comments_from_ts_code(doc.page_content),
        }

        texts.append(doc.page_content)
        metadatas.append(metadata)

    chunks = [
        texts[i : (i + 1000) if (i + 1000) < len(texts) else len(texts)]
        for i in range(0, len(texts), 1000)
    ]
    embeds = []

    print("Have", len(chunks), "chunks")
    print("Last chunk has", len(chunks[-1]), "texts")

    for chunk, i in zip(chunks, range(len(chunks))):
        print("Chunk", i, "of", len(chunk))
        new_embeddings = client.embeddings.create(input=chunk, model=model_name)
        new_embeds = [emb.embedding for emb in new_embeddings.data]

        embeds.extend(new_embeds)

        #  add time sleep if you encounter embedding token rate limit issue
        time.sleep(7)

    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)

    index = pinecone.Index(index_name)

    ids = [str(uuid4()) for _ in range(len(docs))]

    def dict_to_list_of_strings(input_dict):
        result = []
        for key, value in input_dict.items():
            result.append(f"{key}: {value}")
        return result

    vector_type = os.getenv("PROJECT_VECTOR_TYPE") or "PROJECT_VECTOR_TYPE"

    vectors = [
        (
            ids[i],
            embeds[i],
            {
                "text": docs[i].page_content,
                "title": dict_to_list_of_strings(metadatas[i]),
                "vector_type": vector_type,
            },
        )
        for i in range(len(docs))
    ]

    for i in range(0, len(vectors), 100):
        batch = vectors[i : i + 100]
        print("Upserting batch:", i)
        index.upsert(batch)

    print(index.describe_index_stats())


# PROJECTS
"""
Need to separate links because of embedding rate limit, sleep does not work. 
Also need additional uploads. Before uploading additionally, do not forget to remove create/delete index logic from top. 
"""

projects = [
    "https://github.com/rpanic/vale-ui",
    "https://github.com/pico-labs/coinflip-executor-contract",
    "https://github.com/alysiahuggins/proof-of-ownership-zkapp",
    "https://github.com/sausage-dog/minanite",
    "https://github.com/iammadab/dark-chess",
    "https://github.com/gretzke/zkApp-data-types",
    "https://github.com/Sr-santi/mina-ui",
    "https://github.com/Trivo25/offchain-voting-poc",
    "https://github.com/gordonfreemanfree/snarkyjs-ml",
    "https://github.com/chainwayxyz/mCash",
    "https://github.com/mitschabaude/snarkyjs-sudoku",
    "https://github.com/yunus433/snarkyjs-math",
    "https://github.com/RaidasGrisk/zkapp-ui",
    "https://github.com/jackryanservia/wordle",
    "https://github.com/anandcsingh/rankproof",
    "https://github.com/mina-arena/Contracts",
    "https://github.com/zkHumans/zkHumans",
    "https://github.com/zkHumans/zk-kv",
    "https://github.com/racampos/cpone",
    "https://github.com/SutuLabs/MinaCTF",
    "https://github.com/WalletZkApp/zk-keyless-wallet-contracts",
    "https://github.com/AdMetaNetwork/admeta-mina-chort-1",
    "https://github.com/devarend/binance-oracle",
]

# TODO: Have some problem these project
additional_projects = [
    "https://github.com/Identicon-Dao/socialcap",
]

for project in projects:
    parts = project.strip("/").split("/")
    owner, repo = parts[-2], parts[-1]
    project_loader(owner, repo)
    print("Upserted: ", repo, "from", owner)
    time.sleep(1)

"Project Loader Completed!"