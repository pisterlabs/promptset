from dotenv import load_dotenv
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings


load_dotenv()

def load_index():
    pinecone_api_key = os.getenv("PINECONE_TOKEN")
    pinecone.init(api_key=pinecone_api_key, environment="us-west4-gcp")
    return pinecone.Index("phenomena")

def query_index_by_text(text: str):
    """Query the index by text."""

    openai_key = os.getenv("OPENAI_TOKEN")
    embeddings_client = OpenAIEmbeddings(openai_api_key=openai_key, model="text-embedding-ada-002")
    vector = embeddings_client.embed_query(text)

    # get associated efo
    id, vector = get_most_likely_efos(vector, index)
    print(f"The mapped term for {text} is: {id}")
    # get top 20 linked phenos
    phenotypes = get_phenotypes_for_disease(vector, 20, 0, index)
    print(phenotypes)
    for pheno in phenotypes:
        print(pheno["id"], pheno["score"]) # and also print label of id as metadata
    return phenotypes

def get_most_likely_efos(vector, index):
    top_5 = index.query(
            vector=vector,
            top_k=5,
            # i think it makes sense to only assign a disease ID term
            filter={"isDisease": 1, "isPhenotype": 0},
            include_values=True,
            include_metadata=True
    )

    for e in top_5["matches"]:
        if e["score"] > 0.9:
            # add logic to choose efo by suggesting the user to choose from top 5 list
            print(f"Top scoring EFO ID: {e['id']} ({e['score']})")
        return e["id"], e["values"]

def get_phenotypes_for_disease(disease_vector, top_k, threshold, index):
    phenotypes = index.query(
            vector=disease_vector,
            top_k=top_k,
            filter={"isDisease": 0, "isPhenotype": 1},
            include_values=False,
    )
    matches = []
    for pheno in phenotypes["matches"]:
        if pheno["score"] > threshold:
            matches.append(pheno)

    return matches

if __name__ == "__main__":
    index = load_index()

    # get embedding from input text
    usr_input = input("Enter a disease: ")
    vector = query_index_by_text(usr_input)

