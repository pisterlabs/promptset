import langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import GitLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

langchain.verbose = True
langchain.debug = True


def file_filter(file_path):
    return file_path.endswith(".mdx")


if __name__ == "__main__":
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        branch="master",
        file_filter=file_filter,
    )

    raw_docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings()

    query = "AWSのS3からデータを読み込むためのDocumentLoaderはありますか？"

    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)

    result = qa_chain.run(query)
    print(result)