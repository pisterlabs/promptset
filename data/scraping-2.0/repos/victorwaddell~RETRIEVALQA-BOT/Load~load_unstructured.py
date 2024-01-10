from langchain.document_loaders import DirectoryLoader

def load_unstructured(directory):  # Loads data from directory
    loader = DirectoryLoader(directory, show_progress = True)
    files = loader.load()
    print(f"Loaded {len(files)} unstructured files!")
    return  files # Returns list of documents