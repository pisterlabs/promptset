from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.fastembed import FastEmbedEmbeddings
from langchain.vectorstores import FAISS

class VectorStorage:
    def __init__(self):
        self.embed_model = FastEmbedEmbeddings()
        self.vectorstore = None

    def load_url_list(self, urls, headers=None):
        loader = UnstructuredURLLoader(urls=urls, continue_on_failure=False, headers=headers)
        documents = loader.load()
        return documents

    def train_and_save_db(self, urls, file_path, headers=None):
        documents = self.load_url_list(urls, headers=headers)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=950,
            chunk_overlap=90,
            length_function=len,
        )

        documents = text_splitter.split_documents(documents)
        print(len(documents))

        self.vectorstore = FAISS.from_documents(documents, self.embed_model)
        self.vectorstore.save_local(file_path)

    def load_db_and_get_retriever(self, file_path):
        self.vectorstore = FAISS.load_local(file_path, self.embed_model)
        retriever = self.vectorstore.as_retriever()
        return retriever

if __name__ == "__main__":
    
    def read_url_list(file_path):
        with open(file_path, 'r') as file:
            url_list = [line.strip() for line in file.readlines()]
        return url_list    
    
    url_list_file_path = "url_list.txt"
    urls = read_url_list(url_list_file_path)    
    
    # Example usage for testing:
    vector_storage = VectorStorage()

    # Specify headers
    headers = {
        'Referer': 'https://www.google.com/',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
    }

    # Train and save the vector store to a local file
    vector_storage.train_and_save_db(urls, "faiss_index", headers=headers)

    # Load the vector store from the local file and get the retriever
    retriever = vector_storage.load_db_and_get_retriever("faiss_index")
    print("Retriever:", retriever)
