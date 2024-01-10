import os
import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pymongo import MongoClient
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory 
from langchain.vectorstores import MongoDBAtlasVectorSearch
from fastapi.middleware.cors import CORSMiddleware


'''
http://127.0.0.1:8080/docs
Can I find a job in Switzerland as a nurse?
uvicorn chat_fastapi_app:app --reload
'''
#TODO:
# 1. Solve bug pymongo.errors.OperationFailure: 
# $vectorSearch is not allowed or the syntax is incorrect, see the Atlas documentation for more information, full error: {'ok': 0, 'errmsg': '$vectorSearch is not allowed or the syntax is incorrect, 
# see the Atlas documentation for more information', 'code': 8000, 'codeName': 'AtlasError'}
# 2. Add cluster to parse_parameters and use it in MongoDBAtlasVectorSearch
# 3. Simplify code I feel we don't need all these langchain functions
# 4. Write dockerfile and run app in docker
# 5. Think about hosting options

# Load environment variables from the .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Get MongoDB Atlas credentials from environment variables
ATLAS_TOKEN = os.environ["ATLAS_TOKEN"]
ATLAS_USER = os.environ["ATLAS_USER"]

def parse_parameters(start_date, end_date, country, state):
    """
    Parse the input parameters and construct search conditions.
    
    Args:
    - start_date (str): The start date for the date range filter. Defaults to "1999-01-01" if 'null'.
    - end_date (str): The end date for the date range filter. Defaults to "2999-01-01" if 'null'.
    - country (str): The country to filter by. Not used if 'null'.
    - state (str): The state to filter by. Not used if 'null'.
    
    Returns:
    - dict: Constructed search conditions to be used in MongoDB Atlas VectorSearch.

    Sample Usage:
    >>> parse_parameters("2022-01-01", "2022-12-31", "Switzerland", "Zurich")
    {'compound': {'must': [{'text': {'path': 'state', 'query': 'Texas'}}, 
                           {'text': {'path': 'country', 'query': 'USA'}}, 
                           {'range': {'path': 'messageDatetime', 
                                      'gte': datetime.datetime(2022, 1, 1, 0, 0), 
                                      'lte': datetime.datetime(2022, 12, 31, 0, 0)}}]}}
    """
    
    # List to hold our search conditions
    must_conditions = []
    
    # Check and add state condition
    if state != 'null':
        filter = {
            "text": {
                "path": "state",
                "query": state
            }
        }
        must_conditions.append(filter)

    # Check and add country condition
    if country != 'null':
        filter = {
            "text": {
                "path": "country",
                "query": country
            }
        }
        must_conditions.append(filter)

    # Set default start and end dates if not provided
    start_date = '1999-01-01' if start_date == 'null' else start_date
    end_date = '2999-01-01' if end_date == 'null' else end_date

    # Add date range condition
    filter = {
        'range': {
            'path': 'messageDatetime',
            'gte': datetime.datetime.strptime(start_date, "%Y-%m-%d"),
            'lte': datetime.datetime.strptime(end_date, "%Y-%m-%d")+datetime.timedelta(days=1),
        }
    }
    must_conditions.append(filter)

    # Return the constructed conditions
    conditions = {
        "compound": {
            "must": must_conditions
        }
    }
    return conditions

@app.post("/query")
async def query(request: Request):
    """
    Endpoint to process user queries and return relevant answers.
    
    Args:
    - request (Request): FastAPI request object containing query parameters and body.
    
    Returns:
    - dict: A dictionary containing the generated answer and updated chat history.
    
    Sample Usage:
    Using HTTP client or CURL:
    POST /query 
    {
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "country": "Switzerland",
    "state": "Zurich",
    "query": "some query",
    "chat_history": []
    }
    Response:
    {
    "answer": "some answer",
    "chat_history": [["some answer", "some query"]]
    }
    """

    print("Starting /query endpoint...")  # Debug: Indicate the start of the endpoint

    # Get data from the incoming request
    data = await request.json()
    start_date = data.get("start_date", "null")
    end_date = data.get("end_date", "null")
    country = data.get("country", "null")
    state = data.get("state", "null")
    query_text = data.get("query")
    chat_history_list = data.get("chat_history", [])
    print(f"Received Data: Start Date: {start_date}, End Date: {end_date}, Country: {country}, State: {state}, Query: {query_text}")
    
    # Error handling: Ensure a query is provided
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text not provided in the request.")

    # Initialize MongoDB Connection
    print("Initializing MongoDB connection...") 
    client = MongoClient(
        "mongodb+srv://{}:{}@cluster0.fcobsyq.mongodb.net/".format(
            ATLAS_USER, ATLAS_TOKEN))
    collection = client["scrape"]["telegram"]

    # Check for the OpenAI API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not found in environment variables.")

    # Set up embeddings, vectors, and memory for the retrieval chain
    print("Setting up embeddings, vectors, and memory...")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = MongoDBAtlasVectorSearch(
        collection=collection, text_key='messageText',
        embedding=embeddings, index_name='telegram_embedding'
    )

    memory = ConversationBufferMemory( 
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer'
    )

    llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=api_key)
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    Combine the information from the context with your own general knowledge to provide a comprehensive and accurate answer. 
    Please be as specific as possible, also you are a friendly chatbot who is always polite.
    {context}
    Question: {question}"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    # Generate the search conditions and set up the retrieval chain
    must_conditions = parse_parameters(start_date, end_date, country, state)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectors.as_retriever(search_type = 'mmr',
                                       search_kwargs={
                                                'k': 100, 'lambda_mult': 0.25,
                                                "pre_filter": {
                                                   "compound": {
                                                       "must": must_conditions
                                                   }
                                                },
                                       }),
        memory = memory,
        return_source_documents=True,
        return_generated_question=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Process the query using the retrieval chain
    answer = chain({"question": query_text, "chat_history": chat_history_list})

    # # Print details of the source documents
    # print(answer["source_documents"][0].metadata['state'])
    # print(answer["source_documents"][0].metadata['country'])
    # print(answer["source_documents"][0].metadata['messageDatetime'])
    # print(answer["source_documents"][0].page_content)

    # # Add the new Q&A pair to the chat history and return the results
    # print("Returning the response...")
    chat_history_list.append((query_text, answer["answer"]))
    return {"answer": answer["answer"], "chat_history": chat_history_list}

#solely for test & debug
@app.get("/test")
def test_endpoint():
    print("Test endpoint called!")
    return {"message": "Test successful"}


# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)