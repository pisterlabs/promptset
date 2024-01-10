from dotenv import load_dotenv
from langchain.document_loaders import GitLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()

# GitHubからLangChainのリポジトリをクローンし、.mdxファイルを読み込む
# (.mdxファイルはLangChainのドキュメントのもとになるファイルの一部)
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./tmp/langchain",
    branch="master",
    file_filter=lambda f: f.endswith(".mdx"),
)
raw_docs = loader.load()
print(len(raw_docs))

# ドキュメントを1000文字単位で分割（チャンク化）する
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(raw_docs)
print(len(docs))

# ドキュメントをベクトル化して、FAISSに読み込む
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
db.save_local("./tmp/faiss")
