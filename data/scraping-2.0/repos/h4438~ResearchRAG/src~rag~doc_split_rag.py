import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS

class DocSplitRag:
    """Document spliter RAG. 
    # Default Technologies:
    - FAISS
    - RecursiveDocumentSplitter
    - LaBSE

    # Run steps:
    - load major data
    - split data into chunks and attach them with their title
    - embed each chunk with LaBSE model
    - find similar between query with each chunk using Euclid distance (L2) 
    """ 

    def __init__(self, data_path: str):
        self.chunk_size = 1600
        self.chunk_overlap = 200
        self.path = data_path
        self.model = "sentence-transformers/LaBSE"

    def build(self, hug_token:str):
        """
            this method builds the components needed to run this RAG
            @param hug_token hugging face token
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap, length_function=len,
            is_separator_regex=False)
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hug_token, model_name=self.model)
        self._load_data_to_db()
    
    def _load_data_to_db(self):
        """
            This method load text data to vector database
        """
        dist = []
        db = []
        db_df = pd.read_csv(f"{self.path}/majors_info.csv")
        for i, data in db_df.iterrows():
            doc = self.text_splitter.create_documents([data['Description']])
            dist.append(len(doc))
            [d.metadata.update({"doc_title": data['Major']}) for d in doc]
            db.extend(doc)
        self.retriever = FAISS.from_documents(db, self.embeddings)
        return
    
    def get_retriever(self):
        return self.retriever