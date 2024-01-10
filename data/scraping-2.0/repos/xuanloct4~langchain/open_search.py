import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

import os, yaml


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders import TextLoader
from langchain.document_loaders import TextLoader
loader = TextLoader('./documents/state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# docsearch = OpenSearchVectorSearch.from_documents(
#     docs, 
#     embeddings, 
#     opensearch_url="http://localhost:9200"
# )

##If using the default Docker installation, use this instantiation instead:
docsearch = OpenSearchVectorSearch.from_documents(
    docs, 
    embeddings, 
    opensearch_url="https://localhost:9200", 
    http_auth=("admin", "admin"),     
    use_ssl = False,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
)
query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query, k=10)

print(docs[0].page_content)
docsearch = OpenSearchVectorSearch.from_documents(docs, embeddings, 
                                                  opensearch_url="https://localhost:9200",
                                                    http_auth=("admin", "admin"),     
                                                    use_ssl = False,
                                                    verify_certs = False,
                                                    ssl_assert_hostname = False,
                                                    ssl_show_warn = False,
                                                    engine="faiss", space_type="innerproduct", ef_construction=256, m=48)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)

# docsearch = OpenSearchVectorSearch.from_documents(docs, embeddings, opensearch_url="http://localhost:9200", is_appx_search=False)

# query = "What did the president say about Ketanji Brown Jackson"
# docs = docsearch.similarity_search("What did the president say about Ketanji Brown Jackson", k=1, search_type="script_scoring")
# print(docs[0].page_content)


# docsearch = OpenSearchVectorSearch.from_documents(docs, embeddings, opensearch_url="http://localhost:9200", is_appx_search=False)
# filter = {"bool": {"filter": {"term": {"text": "smuggling"}}}}
# query = "What did the president say about Ketanji Brown Jackson"
# docs = docsearch.similarity_search("What did the president say about Ketanji Brown Jackson", search_type="painless_scripting", space_type="cosineSimilarity", pre_filter=filter)
# print(docs[0].page_content)


# # this is just an example, you would need to change these values to point to another opensearch instance
# docsearch = OpenSearchVectorSearch(index_name="index-*", embedding_function=embeddings, opensearch_url="http://localhost:9200")

# # you can specify custom field names to match the fields you're using to store your embedding, document text value, and metadata
# docs = docsearch.similarity_search("Who was asking about getting lunch today?", search_type="script_scoring", space_type="cosinesimil", vector_field="message_embedding", text_field="message", metadata_field="message_metadata")