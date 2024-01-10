from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from utility import pickle_object
import os


def load_data():
    txt_loader = DirectoryLoader(
        "./data/", glob="**/*.txt", show_progress=True, use_multithreading=True
    )
    py_loader = DirectoryLoader(
        "./data/", glob="**/*.py", show_progress=True, use_multithreading=True
    )

    js_loader = DirectoryLoader(
        "./data/", glob="**/*.js", show_progress=True, use_multithreading=True
    )

    ts_loader = DirectoryLoader(
        "./data/", glob="**/*.ts", show_progress=True, use_multithreading=True
    )

    tsx_loader = DirectoryLoader(
        "./data/", glob="**/*.tsx", show_progress=True, use_multithreading=True
    )

    java_loader = DirectoryLoader(
        "./data/", glob="**/*.java", show_progress=True, use_multithreading=True
    )

    kt_loader = DirectoryLoader(
        "./data/", glob="**/*.kt", show_progress=True, use_multithreading=True
    )

    loaders = [txt_loader, py_loader, js_loader, ts_loader, tsx_loader, java_loader, kt_loader]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    print(f"Total number of documents: {len(documents)}")
    return documents


def split_data(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    return documents


if __name__ == "__main__":
    documents = load_data()
    documents = split_data(documents)

    if not os.path.exists("output"):
        os.mkdir("output")
    pickle_object(documents, "output/documents.bin")
    print("saved to output/documents.bin")
