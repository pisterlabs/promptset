from langchain.document_loaders.generic import GenericLoader
from langchain.text_splitter import Language
from langchain.document_loaders.parsers import LanguageParser
import os

def find_java_directories(repo_path):
    """
    Recursively find directories containing Java files.
    """
    java_directories = set()
    for root, dirs, files in os.walk(repo_path):
        if any(file.endswith(".java") for file in files):
            java_directories.add(root)
    return list(java_directories)

def remove_duplicate_documents(documents):
    """
    Remove duplicate documents based on their source file path.
    """
    unique_docs = {}
    for doc in documents:
        source_path = doc.metadata.get('source')
        if source_path and source_path not in unique_docs:
            unique_docs[source_path] = doc
    return list(unique_docs.values())

def count_java_files(directory):
    """
    Count the number of Java files in a given directory.
    """
    return sum(1 for file in os.listdir(directory) if file.endswith(".java"))

def load_java_documents_from_repo(repo_path):
    """
    Load all Java documents from the specified repository path.
    """
    java_directories = find_java_directories(repo_path)
    documents = []
    for java_dir in java_directories:
        loader = GenericLoader.from_filesystem(
            path=java_dir,
            glob="**/*.java",
            suffixes=[".java"],
            parser=LanguageParser(language=Language.JAVA, parser_threshold=500)
        )
        documents.extend(loader.load())
    
    return remove_duplicate_documents(documents)

def main():
    # Replace with the actual repository path after cloning
    repo_path = "/path/to/cloned/repository"
    documents = load_java_documents_from_repo(repo_path)

    print(f"Total number of unique documents: {len(documents)}")
    for doc in documents:
        print(f"Document Source: {doc.metadata['source']}")

if __name__ == "__main__":
    main()
