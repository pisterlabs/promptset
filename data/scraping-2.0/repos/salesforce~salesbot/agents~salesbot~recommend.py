import os
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


class RecommendModule:
    def __init__(self, verbose=False):
        index_name = "products_faiss_index"
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        if os.path.isdir(index_name):
            print(f"load local {index_name}")
            self.db = FAISS.load_local(index_name, embeddings)
        else:
            datapath = "data/products/"
            names = [datapath+d for d in os.listdir(datapath)]
            products = []
            for fpath in names:
                print(fpath)
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    for k,values in data.items():
                        for v in values:
                            v['category'] = k
                            v['price_float'] = float(v['price'].replace('$','').replace(',',''))
                            products.append(v)

            docs = []
            for i, product in enumerate(products):
                title = product['name'].strip()
                price = product['price'].strip()
                description = product['description']
                feature = ', '.join(product['features'])
                product_doc = f"{title}\nPrice: {price}\n{description}\n{feature}"
                contents = f"Name: {title}\nPrice: {price}\nDescription: {description}\nFeatures: {feature}"
                docs.append(Document(
                    page_content=product_doc,
                    metadata={'title': title, 'id': str(i), 'contents': contents}))
            print(f"Processed {len(docs)} docs")

            self.db = FAISS.from_documents(docs, embeddings)
            self.db.save_local(index_name)
        print("Loaded product db")

    def top_docs(self, query: str, k: int = 4):
        top_documents = self.db.similarity_search(query, k=k)
        return top_documents
