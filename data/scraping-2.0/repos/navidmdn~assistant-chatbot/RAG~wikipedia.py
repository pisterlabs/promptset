from langchain.retrievers import WikipediaRetriever

retriever = WikipediaRetriever()
docs = retriever.get_relevant_documents(query="HUNTER X HUNTER")
print(docs[0].page_content[:100])
