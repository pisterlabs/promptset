from typing import List
from itertools import chain

from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.docstore.document import Document


def load_website(url: str, exclude_dirs: List[str] = None) -> List[Document]:
    """
    Given a url, returns the website loaded as langchain documents.
    """
    return RecursiveUrlLoader(
        url=url,
        exclude_dirs=exclude_dirs,
    ).load()


def load_weblinks(weblinks_file_path: str):
    """
    Given a file path to a text file with a list of urls, returns the websites loaded as documents.
    """
    with open(weblinks_file_path, "r") as f:
        urls = f.readlines()
    return list(chain.from_iterable((load_website(url) for url in urls)))


if __name__ == "__main__":
    documents = load_weblinks(
        "C:\\Users\\Hendrix\\Documents\\GitHub\\strategy-ai\\backend\\strategy_ai\\available_data\\visible_files\\weblinks.txt")
    print(type(documents))
    print(type(documents[0]))
    print(documents)

    # documents = DocStore(dict({
    #     "Website Documents": DocumentSource(name="Website Documents", filePaths=[os.path.join(available_documents_directory, "visible_files", "weblinks.txt")]),
    # }))

    # vectorStore = FAISSVectorStore(documents.splitDocuments)

    # llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)

    # llm.predict_messages()
