from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from client import persistent_client, embeddings

def filter_relevant_pdfs(query):    
    langchain_chroma_pdf = Chroma(
        client=persistent_client,
        collection_name="pdf",
        embedding_function=embeddings,
    )
    smaller_chunk_pdf = Chroma(
        client=persistent_client,
        collection_name="pdf_child",
        embedding_function=embeddings,
    )
    small_len = persistent_client.get_or_create_collection(name="pdf_child", embedding_function=embeddings).count()
    filtered_docs = smaller_chunk_pdf.similarity_search_with_score(query,small_len)

    threshold = 1
    filtered_docs = list(filter(lambda x: x[1] < threshold, filtered_docs))

    print(f"Child Chunks Num: {len(filtered_docs)}")

    papers = []
    parentIDs = []
    for doc in filtered_docs:
        papers.append(doc[0].metadata['fileName'])
        parentIDs.append(doc[0].metadata['parentID'])
    

    filtered_docs.sort(key = lambda x: x[1] )
    papers = list(dict.fromkeys(papers))
    parentIDs = list(dict.fromkeys(parentIDs))

    relevant_chunks = langchain_chroma_pdf.get(ids=parentIDs)
    print(f"Parent Chunks Num: {len(relevant_chunks['ids'])}")

    try:
        persistent_client.delete_collection(name="pdf_relevant")
    except:
        print('no pdf collection')

    relevant_chunks_collection = persistent_client.get_or_create_collection(name="pdf_relevant")
    relevant_chunks_collection.add(ids = relevant_chunks['ids'],
                                metadatas = relevant_chunks['metadatas'],
                                documents = relevant_chunks['documents'],
                                embeddings = relevant_chunks['embeddings'])
    return papers


## TEST
# query ="cultural application of pychological first-aid in various countries"
# relevant_papers = filter_relevant_pdfs(query)

# relevant_chunks_collection = persistent_client.get_collection(name="pdf_relevant")

# print(relevant_chunks_collection.peek())
# print(relevant_chunks_collection.get(
#     where={"fileName": "data\\92\\Effectiveness of psychological first aid training for social work students, practitioners and human service professionals in Alberta, Canada.pdf"}
# ))

# print("Done")


# print(persistent_client.get_or_create_collection(name="pdf_child").get())