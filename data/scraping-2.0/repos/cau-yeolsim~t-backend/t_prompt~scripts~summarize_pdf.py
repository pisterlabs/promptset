import os

from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.schema.document import Document

load_dotenv()


def get_chunked_docs(docs: list[Document], chunk_size: int = 10_000) -> list[Document]:
    text_count = 0
    next_docs = []
    next_docs_idx_list = []
    for idx, doc in enumerate(docs):
        text_count += len(doc.page_content)
        if text_count > chunk_size:
            yield next_docs_idx_list, next_docs
            text_count = 0
            next_docs_idx_list = [idx]
            next_docs = [doc]
        else:
            next_docs_idx_list.append(idx)
            next_docs.append(doc)


def summarize(file_path: str):
    print(file_path + " 요약 시작.", flush=True)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-16k",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
    chain = load_summarize_chain(llm, chain_type="stuff")
    summarized_text = ""
    for idx_list, chunk in get_chunked_docs(docs):
        print(str(idx_list) + "chunk 에 대한 요약 시작.", flush=True)
        summarized_text += chain.run(chunk) + "\n\n"

    summary_file_path = file_path.replace(".pdf", "_summary.txt")
    with open(summary_file_path, "w") as f:
        f.write(summarized_text)
    print(file_path + " 요약 완료.", flush=True)


def get_pdf_file_paths(root_dir: str) -> list[str]:
    files = os.listdir(root_dir)
    pdf_file_paths: list[str] = []
    for file in files:
        path = os.path.join(root_dir, file)
        if path.endswith(".pdf"):
            pdf_file_paths.append(path)
    return pdf_file_paths


def load_pdf(root_dir: str = "/Users/shmoon/Desktop/papers"):
    pdf_file_paths = get_pdf_file_paths(root_dir)
    for path in pdf_file_paths[4:]:
        summarize(path)


if __name__ == "__main__":
    load_pdf()
