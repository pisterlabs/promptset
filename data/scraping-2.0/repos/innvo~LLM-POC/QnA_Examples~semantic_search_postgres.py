from langchain.embeddings.openai import OpenAIEmbeddings
import os
import psycopg2

# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')

## Set local environment variables 
OPENAI_API_KEY=os.getenv("OPEN_API_KEY")
db_user=os.getenv("DBUSER")
db_password=os.getenv("DBPASSWORD")
db_host=os.getenv("DBHOST")

embeddings = OpenAIEmbeddings()

## search emebedding
def search_embedding(query_string):
    print("in search_embedding")
    query_embedding = embeddings.embed_query(query_string)
    return query_embedding

## similarity_search
def similarity_search(query_string):
    print("in similarity_search")
    query_embedding = search_embedding(query_string)
    # print("query_embedding: " + str(query_embedding))
    conn = psycopg2.connect(
        dbname="llm-demo",
        user=db_user,
        password=db_password,
        host=db_host,
        port="5432"
        )
    # Open a cursor to perform database operations
    cur = conn.cursor()
    # Execute aquery
    cur.execute(f"""
    select * from (
        select id, left(embedding_content, 255), 1 - (embedding_vector <=> '{query_embedding}') as cosine_similarity 
         from public.contentembedding
        ) subquery
    where cosine_similarity > 0.77
    order by cosine_similarity desc
    """)
    # Fetch all resultscur.execute(f
    results = cur.fetchall()
    # Print results
    for result in results:
        print(result)
    # Commit changes
    conn.commit()
    # Close cursor and connection
    cur.close()
    conn.close()
  
###############################
query_string =  "What did the president say about Justice Breyer" 
similarity_search(query_string)

## No Match
#query_string = "What did the president say about Justice Breyer" 
## Exact Match
# query_string ="to exploitative working conditions such as working excessive hours or having their wages withheld mainly in domestic labor page 26"
# query_string ="what types of abuses do you see"

