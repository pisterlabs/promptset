import argparse
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# create class for querying the vector DB
class QueryDB:
    def __init__(self, model):
        self.model = model
        self.load_db()

    # create vector DB retriever
    def load_db(self):
        try:
            hf = HuggingFaceEmbeddings(model_name=self.model)

            vector_index_dir = "db"
            self.vectordb = Chroma(persist_directory=vector_index_dir, embedding_function=hf)

            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 2})
        except Exception as e:
            print("An error occurred while loading the DB:", str(e))
        
    def query_db(self, query):
        try:
            docs = self.retriever.get_relevant_documents(query)

            print("Query: ", query, "\n\nSources and Contents: ")
            for doc in docs:
                print("\nSource:", doc.metadata["source"])
                print("Content:", doc.page_content, "\n\n")
        except Exception as e:
            print("An error occurred while querying the DB:", str(e))



# load the model once and use it
model_name = "sentence-transformers/all-mpnet-base-v2"
query_db = QueryDB(model_name)

def main(query):
    query_db.query_db(query)


# only instantiate the class if this file is run directly
if __name__ == "__main__":
    
    #create parser for obtaining commandline query from user
    parser = argparse.ArgumentParser(description='Query the vector DB')
    parser.add_argument('query', metavar='query', type=str, nargs='+',
                        help='query to search for in the vector DB')
    args = parser.parse_args()
    query = " ".join(args.query)

    main(query)
