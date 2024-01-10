import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp(u"invested")

for token in doc:
    print(token, token.lemma, token.lemma_)
# Load environment variables
load_dotenv()

# Initialize the embeddings model
embedding_model = "multi-qa-MiniLM-L6-cos-v1"
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

vector_db_relations = Milvus(
    collection_name="RelationCollection",
    embedding_function=HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1"),
    connection_args={
            "uri": os.getenv("MILVUS_URI"),
            "token": os.getenv("MILVUS_TOKEN"),
            "secure": True,
        },
)

def search_in_vector_db(query, top_k=10):
    """
    Search for the top K matches in the Milvus vector database for the given query.

    :param query: The query string to search for.
    :param top_k: Number of top matches to return.
    :return: A list of top K matches with similarity scores.
    """
    # Generate embedding for the query
    query_embedding = embedding_function.embed_query(query)
    #print(query_embedding)
    # Perform the similarity search
    search_results = vector_db_relations.similarity_search(query, top_k=top_k)
    print(search_results)
    # Process and return the results
    matches = []
    for result in search_results:
        """ match = {
            "id": result.id,
            "score": result.score
        }
        matches.append(match) """
    return matches

# Example usage
query_string = "(Organization, invest, Person)'"
top_matches = search_in_vector_db(query_string)
