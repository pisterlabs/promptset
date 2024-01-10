# This version simply uses a recipe dataset from hugging face

# Update these with your own api keys and paths

ASTRA_DB_SECURE_BUNDLE_PATH="your secure bundle path here"
ASTRA_DB_APPLICATION_TOKEN="Your token here"
ASTRA_DB_CLIENT_ID="Your client id here"
ASTRA_DB_CLIENT_SECRET ="Your client secret here"
ASTRA_DB_KEYSPACE="search"
OPENAI_API_KEY="Your openai api key here"

from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

cloud_config= {
        'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

myCassandraVStore = Cassandra(
    embedding=myEmbedding,
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="recipe_db_demo",
)

print("Loading data from huggingface")
dataset = load_dataset("corbt/unlabeled-recipes", split="train")
headlines = dataset["recipe"][:100]

print("\nGenerating embedding and storing in AstraDB")
myCassandraVStore.add_texts(headlines)

print("Inserted %i recipes.\n" % len(headlines))


vectorIdx = VectorStoreIndexWrapper(vectorstore=myCassandraVStore)

first_question = True
while True:
    if first_question:
        query_text = input("Ask me a for a recipe: (or type 'quit to exit): ")

        first_question = False
    else:
        query_text = input("Ask me a for a recipe: (or type 'quit to exit): ")


    if query_text == "quit":
        break

    print("QUESTION: \%s\"" % query_text)
    answer = vectorIdx.query(query_text, llm=llm).strip()
    # print("ANSWER: \"%s\"\n" % answer)

    print("Recipes BY RELEVANCE:")
    for doc, score in myCassandraVStore.similarity_search_with_score(query_text, k=4):
        print("  %0.4f \"%s...\"" % (score, doc.page_content[:60]))
        