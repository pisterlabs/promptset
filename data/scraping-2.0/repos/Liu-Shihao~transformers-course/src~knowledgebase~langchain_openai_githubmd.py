import os
import requests
import fnmatch
import argparse
import base64

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

GITHUB_TOKEN = "[INSERT YOUR GITHUB ACCESS TOKEN HERE]"

"""

创建一个 argparse.ArgumentParser 实例来解析命令行参数并获取 GitHub 仓库 URL。
调用 parse_github_url(args.url) 从 URL 中提取仓库所有者和名称。
使用 get_files_from_github_repo(GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN) 从 GitHub 仓库获取所有文件。
定义 Chroma 数据库的本地存储路径，该数据库将存储处理后的文本块。检查 Chroma 数据库是否已经存在。如果不存在，则使用处理后的文本块和 OpenAIEmbeddings 创建一个。如果已经存在，则加载现有数据库。
使用 load_qa_chain(OpenAI(temperature=1), chain_type="stuff") 从 LangChain 库加载预训练的问答链。
创建一个 RetrievalQA 实例，将问答链与 Chroma 数据库相结合。h. 持续提示用户提出问题，并为每个问题使用 qa.run(user_input) 方法根据 Markdown 文件中的信息生成答案。

"""
def parse_github_url(url):
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo


def get_files_from_github_repo(owner, repo, token):
    url = f"<https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1>"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content["tree"]
    else:
        raise ValueError(f"Error fetching repo contents: {response.status_code}")


def fetch_md_contents(files):
    md_contents = []
    for file in files:
        if file["type"] == "blob" and fnmatch.fnmatch(file["path"], "*.md"):
            response = requests.get(file["url"])
            if response.status_code == 200:
                content = response.json()["content"]
                decoded_content = base64.b64decode(content).decode('utf-8')
                print("Fetching Content from ", file['path'])
                md_contents.append(Document(page_content=decoded_content, metadata={"source": file['path']}))
            else:
                print(f"Error downloading file {file['path']}: {response.status_code}")
    return md_contents


def get_source_chunks(files):
    print("In get_source_chunks ...")
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in fetch_md_contents(files):
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadate=source.metadata))
    return source_chunks


def main():
    parser = argparse.ArgumentParser(description="Fetch all *.md files from a GitHub repository.")
    parser.add_argument("url", help="GitHub repository URL")
    args = parser.parse_args()

    GITHUB_OWNER, GITHUB_REPO = parse_github_url(args.url)

    all_files = get_files_from_github_repo(GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN)

    CHROMA_DB_PATH = f'./chroma/{os.path.basename(GITHUB_REPO)}'

    chroma_db = None

    if not os.path.exists(CHROMA_DB_PATH):
        print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
        source_chunks = get_source_chunks(all_files)
        chroma_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
        chroma_db.persist()
    else:
        print(f'Loading Chroma DB from {CHROMA_DB_PATH} ... ')
        chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())

    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=chroma_db.as_retriever())

    while True:
        print('\\n\\n\\033[31m' + 'Ask a question' + '\\033[m')
        user_input = input()
        print('\\033[31m' + qa.run(user_input) + '\\033[m')


if __name__ == "__main__":
    main()