from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def load_pdf_document(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def initialize_embeddings():
    return OpenAIEmbeddings()

def store_documents_in_vector_store(docs, embeddings):
    db = Chroma.from_documents(docs, embeddings)
    return db.as_retriever()

def retrieve_documents(retriever, retrieval_query):
    return retriever.get_relevant_documents(retrieval_query)

def initialize_qa_chain(retriever, model_name="gpt-4-1106-preview"):
    chat = ChatOpenAI(model_name=model_name)
    return RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)

# メインの処理
if __name__ == "__main__":
    file_path = "path_to_your_pdf_file.pdf"
    retrieval_query = "Exercisesを抽出してください。"
    qa_query = "Exerciseを1題選択し日本語に翻訳してください。"

    # PDFを読み込む
    pages = load_pdf_document(file_path)

    # ドキュメントをチャンクに分割
    docs = split_documents_into_chunks(pages)

    # 埋め込みモデルの初期化
    embeddings = initialize_embeddings()

    # ベクトルストアにドキュメントを格納
    retriever = store_documents_in_vector_store(docs, embeddings)

    # ドキュメントを抽出
    context_docs = retrieve_documents(retriever, retrieval_query)
    print(f"len = {len(context_docs)}") # 抽出したドキュメントの数

    first_doc = context_docs[0] # 1つ目のドキュメント
    print(f"metadata = {first_doc.metadata}") # 1つ目のドキュメントのメタデータ
    print(first_doc.page_content) # 1つ目のドキュメントの中身

    # QAチェーンの初期化と実行
    qa_chain = initialize_qa_chain(retriever)
    result = qa_chain.run(qa_query)
    print(result)
