# ./vector/chroma_threads.py
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from credentials import oai_api_key
from configuration import vector_folder_path
from database.confluence_database import get_page_data_from_db
from database.confluence_database import update_embed_date


def vectorize_documents(all_documents, page_ids):
    """
    Vectorize a list of documents and add them to the vectorstore.
    :param all_documents:
    :param page_ids:
    :return: page ids of the vectorized documents
    """

    # Initialize OpenAI embeddings with the API key
    embedding = OpenAIEmbeddings(openai_api_key=oai_api_key)

    # Create the Chroma vectorstore with the embedding function
    vectordb = Chroma(embedding_function=embedding, persist_directory=vector_folder_path)

    # Prepare page_ids to be added as metadata
    metadatas = [{"page_id": page_id} for page_id in page_ids]

    # Add texts to the vectorstore
    vectordb.add_texts(texts=all_documents, metadatas=metadatas)

    # Persist the database
    vectordb.persist()

    # Update the last_embedded timestamp in the database
    update_embed_date(page_ids)

    # Return the page ids of the vectorized documents
    return page_ids


def add_to_vector():
    """
    Vectorize all new or updated documents and add them to the vectorstore.
    :return: page ids
    """
    all_documents, page_ids = get_page_data_from_db()

    # Check if the lists are empty
    if not all_documents or not page_ids:
        print("No new or updated documents to vectorize.")
        return []

    vectorize_documents(all_documents, page_ids)
    print(f'Vectorized {len(all_documents)} documents.')
    print(f'Vectorized page ids: {page_ids}')
    return page_ids


def retrieve_relevant_documents(question):
    """
    Retrieve the most relevant documents for a given question.
    :param question:
    :return: document ids
    """
    # Initialize OpenAI embeddings with the API key
    embedding = OpenAIEmbeddings(openai_api_key=oai_api_key)

    # Create the Chroma vectorstore with the embedding function
    vectordb = Chroma(embedding_function=embedding, persist_directory=vector_folder_path)

    # Embed the query text using the embedding function
    query_embedding = embedding.embed_query(question)

    # Perform a similarity search in the vectorstore
    similar_documents = vectordb.similarity_search_by_vector(query_embedding)

    # Process and return the results along with their metadata
    results = []
    for doc in similar_documents:
        result = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        results.append(result)
    document_ids = [doc.metadata.get('page_id') for doc in similar_documents if doc.metadata]

    return document_ids


def retrieve_relevant_documents_with_proximity(question, max_proximity=0.5, max_results=10):
    """
    Retrieve the most relevant documents for a given question, filtering them by maximum proximity.
    :param question: The query text.
    :param max_proximity: The maximum proximity value for filtering documents.
    :param max_results: The maximum number of results to return.
    :return: A list of tuples containing document ids and their proximity values.
    """
    # Initialize OpenAI embeddings with the API key
    embedding = OpenAIEmbeddings(openai_api_key=oai_api_key)

    # Create the Chroma vectorstore with the embedding function
    vectordb = Chroma(embedding_function=embedding, persist_directory=vector_folder_path)

    # Embed the query text using the embedding function
    query_embedding = embedding.embed_query(question)

    # Perform a similarity search in the vectorstore
    similar_documents_with_scores = vectordb.similarity_search_by_vector_with_relevance_scores(
        query_embedding, k=max_results
    )

    # Process the results, filtering by the maximum proximity value
    filtered_results = []
    for doc, score in similar_documents_with_scores:
        if score <= max_proximity:
            result = {
                "page_id": doc.metadata.get('page_id'),
                "proximity_value": score
            }
            filtered_results.append(result)

    return filtered_results



if __name__ == '__main__':
    # vectorized_page_ids = add_to_vector()
    question = "what is the functionality of this solution?"
    relevant_document_ids = retrieve_relevant_documents_with_proximity(question)
    for result in relevant_document_ids:
        print(result)
        print("---------------------------------------------------")