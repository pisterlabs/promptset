import json
import os
from langchain_core.documents import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma


class FunctionMatcher:
    def __init__(self):
        # load functions config file
        self.functions = json.load(
            open(os.path.join(os.getcwd(), "Config", "functions.json"))
        )

        docs = []
        for f in self.functions:
            for p in f["possible_phrases"]:
                docs.append(Document(page_content=p, metadata={"function": f["name"]}))
        # create the open-source embedding function
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.db = Chroma(embedding_function=self.embedding_function)
        self.db.add_documents(docs)

        print(f"Totally {self.db.get()} entries inserted in the chromadb")

        self.TRESHOLD = 0.2

    def get_similar(self, user_prompt=""):
        results = self.db.similarity_search_with_relevance_scores(
            query=user_prompt, k=1, score_threshold=self.TRESHOLD
        )
        if len(results) == 1:
            print(results)
            return results[0][0].metadata["function"]
        return None
