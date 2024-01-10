from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import openai, json

query = "tender topic, submission data, submission address & earnest money"

functions= [  
    {
        "name": "readPDF",
        "description": "reads PDF and returns the content of the pages that have the information for the requested query",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Search Query parameters for the PDF file. If multiple data points requested then list all of them comma separated."
                }
            },
            "required": ["search_query"]
        }
    }
]

def readPDF(search_query):
    loader = PyPDFLoader("tender.pdf")
    pages = loader.load_and_split()
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings(deployment="ada-002", model="text-embedding-ada-002",retry_min_seconds=30, disallowed_special=()))
    print (faiss_index.index.ntotal)
    query = search_query.split(",")
    search_content = ""
    for q in query:
        docs = faiss_index.similarity_search(q, k=2)
        for doc in docs:
            search_content = search_content + doc.page_content
            #print("\n Page No :: " + str(doc.metadata["page"]) + ": \n", doc.page_content)
        #print (search_content)
    return search_content

system_message = """You are an assistant designed to help people understand tender documents specifics from a tender PDF file uploaded.
The user may ask for multiple data points in the tender document and you will identify each of the data points and try to answer all of them.
You have access to a PDF reader that can read the tender document and extract the relevant information.
Make sure the answer is detailed and relevant and is not a very brief summary.
"""
messages= [{"role": "system", "content": system_message},
    {"role": "user", "content": query},
]

openai.api_key=os.getenv("OPENAI_API_KEY")  
openai.api_version="2023-10-01-preview"
api_model=os.getenv("OPENAI_API_MODEL")
openai.api_type="azure"
openai.api_base = os.getenv("OPENAI_API_URL")

response = openai.ChatCompletion.create(
    engine=api_model,
    messages= messages,
    functions = functions,
    function_call="auto",
)

output = response.choices[0].message
print(output)

if "function_call" in output:
    function_name = output["function_call"]["name"]
    function_arguments = json.loads(output["function_call"].arguments)
    available_functions = {
            "readPDF": readPDF,
    }
    function_to_call = available_functions[function_name]
    print(function_arguments)
    function_response = function_to_call(**function_arguments)

    messages.append(
        {"role": output["role"], 
         "function_call":{
             "name": output["function_call"].name, 
             "arguments": output["function_call"].arguments }, 
             "content": None})
    messages.append({"role": "function", "name" : function_name, "content": function_response})

    response = openai.ChatCompletion.create(
        engine=api_model,
        messages= messages,
        functions = functions,
        function_call="auto",
    )
    output = response.choices[0].message
    print(output)


