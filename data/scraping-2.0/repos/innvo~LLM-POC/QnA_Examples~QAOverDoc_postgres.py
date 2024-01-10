from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
import langchain
import os
import psycopg2

# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')

embeddings = OpenAIEmbeddings()
## Set local environment variables
folder_path = "QnA/country_reports/content"
OPENAI_API_KEY=os.getenv("OPEN_API_KEY")
db_user=os.getenv("DBUSER")
db_password=os.getenv("DBPASSWORD")
db_host=os.getenv("DBHOST")

## Langchain setup
model = langchain.OpenAI(temperature=0, model_name="gpt-4")

embeddings = OpenAIEmbeddings()

## search emebedding
def search_embedding(query_string):
    print("in search_embedding")
    query_embedding = embeddings.embed_query(query_string)
    return query_embedding

## similarity_search
def get_documents(question):
    print("in similarity_search")
    docs= []
    query_embedding = search_embedding(question)
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
    cur.execute(f""" select id, embedding_content, 
                 1 - (embedding_vector <=> '{query_embedding}') as cosine_similarity 
                 from public.contentembedding 
                 order by 3 desc
                 limit 20""") 
    # Fetch all results
    results = cur.fetchall()

    # Close cursor and connection
    cur.close()
    conn.close()

    # Create docs list for langchain Qa Chain
    for result in results:   
       doc =Document(
           page_content=result[1]
       )
       docs.append(doc)     
    get_response_from_llm(docs)
  
## Get response from langchain Qa Chain   
def get_response_from_llm(docs):
    # Load QA Chain
    qa_chain = load_qa_chain(model, chain_type="stuff")
    response = qa_chain.run(
        question=question, 
        input_documents=docs
    ) 
    print(response)

## Generate the query embedding 
def answer_question(question):
    get_documents(question)


###############################
#question =  "What did the president say about Justice Breyer" 
#question =  "What did the president say about immigration. Provide 5 as bullets.  be concise"   
question =  "What did the president Biden say about southern border. Provide 3 as bullets.  be concise"
#question = "What are the top 5 topics discussed by president biden"
#question = "What is the president' birthday"

answer_question(question)

# Quickstart
