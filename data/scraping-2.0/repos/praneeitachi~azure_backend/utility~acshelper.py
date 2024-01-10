# Import required libraries
import os
import openai
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import Vector
import ast
from azure.storage.blob import BlobServiceClient


# Configure environment variables
load_dotenv("acs.env")

service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
qa_index_name = os.getenv("AZURE_QA_INDEX_NAME")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
azure_search_credential = AzureKeyCredential(key)
chunk_size = int(os.environ["chunk_size"])
similar_chunk_count = os.environ["similar_chunk_count"]
k = int(similar_chunk_count)


EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME_1")
GPT3_LLM_MODEL_DEPLOYMENT_NAME = os.environ["GPT3_LLM_MODEL_DEPLOYMENT_NAME"]
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

blob_connection_string = os.getenv("BLOB_STORAGE_CONNECTION_STRING")
blob_container_name = os.getenv("BLOB_STORAGE_CONTAINER_NAME")





#----------------------- INDEXING START#------------------------------------

# # Configure logging
# logging.basicConfig(level=logging.INFO)  # Set the desired logging level
# # Define a logger
# logger = logging.getLogger(__name__)


search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=azure_search_credential)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):
    try:
        # logger.info("Entered embedding function ^^^^^^^^^^^^^^^^^^^^^ ****************")
        # logger.info(f"Starting embedding of:*** {text}  *** with type {type(text)}")
        response = openai.Embedding.create(
            input=text, engine=EMBEDDING_MODEL_DEPLOYMENT_NAME)
        # logger.info("Response returned from azure openAi embedd****************")
        embeddings = response['data'][0]['embedding']
        # logger.info(response['usage']['total_tokens'])
        print(response['usage']['total_tokens'])
        # logger.info(response)
        # print(response)
        # logger.info("Azure OpenAI API call successful")
        return embeddings
    except openai.error.OpenAIError as e:
        # Handle OpenAI API errors
        # logger.warning(f"Error calling OpenAI API:{e}", exc_info=True)
        raise e
    except Exception as e:
        # Handle other exceptions
        # logger.error(f"An unexpected error occurred:{e}", exc_info=True)
        raise e

#---------------- QUERYING---------------------------
def generate_query_embeddings(text):
    try:
        # logger.info("Entered embedding function ^^^^^^^^^^^^^^^^^^^^^ ****************")
        # logger.info(f"Starting embedding of:*** {text}  *** with type {type(text)}")
        response = openai.Embedding.create(
            input=text, engine=EMBEDDING_MODEL_DEPLOYMENT_NAME)
        # logger.info("Response returned from azure openAi embedd****************")
        embeddings = response['data'][0]['embedding']
        # logger.info("Azure OpenAI API call successful")
        return embeddings
    except openai.error.OpenAIError as e:
        # Handle OpenAI API errors
        # logger.warning(f"Error calling OpenAI API:{e}", exc_info=True)
        raise e
    except Exception as e:
        # Handle other exceptions
        # logger.error(f"An unexpected error occurred:{e}", exc_info=True)
        raise e


def get_relevant_chunks(query,k):
    search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=azure_search_credential)

    results = search_client.search(
        search_text=query,
        vector=Vector(value=generate_query_embeddings(query), k=k, fields="contentVector"),
        select=["title", "content"],
        top=k
    )
    # context = " ".join([result['content'] for result in results])
    # results = list(results)
    context = {result['content']:result['title'] for result in results} # because title can be same that's why not made it key
    tracing = []
    for key in context:
        tracing.append({'file_name':context[key],'file_content':key})
    # tracing = { 'file_name': result['title'], 'file_content': result['content'] for result in results}
    # print(len(context))
    return context,tracing


def acs_query(query, category_id, temperature, nochunks):


    # temperature=0
    # k=4
    max_tokens=1024

    # logger.info(f"Query ---------: {query}")

    template = """
    I will ask you questions based on the following context:
    — Start of Context —
    {context}
    — End of Context—
    Use the information in the above paragraphs only to answer the question at the end. If the answer is not given in the context, say that "I do not know".

    Question: {question}

    Response:
    """

    context,tracing = get_relevant_chunks(query, nochunks)
    # logger.info(f"context from chunks ******: {context}")

    prompt = template.format(context=context, question=query)

    response = openai.ChatCompletion.create(
        engine = GPT3_LLM_MODEL_DEPLOYMENT_NAME,
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{prompt}"}
        ],
        max_tokens = max_tokens,
        n = 1,  # specifies the number of alternative completions or responses you want the model to generate for a given prompt
        stop = None,
        temperature = temperature,
        )

    answer = response.choices[0].message["content"]
    token_used = response['usage']['total_tokens']
    print(f"Response: {response}\n")
    print(f"Question: {query}\n")
    print(f"Answer: {answer}\n")
    print(f"Tracing:\n {tracing}")
    return answer, token_used, tracing

#QNA INDEX
def chunking_query(item):
    input_data = []
    text = item['question']
    # Divide text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), 50)]

    # Create items for each chunk
    for i, chunk in enumerate(chunks):
        item_chunk = {
            'id': f"{item['id']}_{i+1}",
            'question':chunk,
            'answer':item['id'],
            'category_id': str(item['category_id']),
            'index_format':item['index_format'],
            'cost_saved':str(item['cost_saved']),
            'tracing':item['tracing']
        }
        input_data.append(item_chunk)

    return input_data   

def qna_indexing(id,question,answer,thoughts,data_points, exclude_category):
    try:
        qa_client = SearchClient(endpoint=service_endpoint, index_name=qa_index_name, credential=azure_search_credential)
        item = {
            'id': id,
            'question':question,
            'answer':answer,
            'exclude_category':str(exclude_category),
            'thoughts':str(thoughts),
            'data_points':data_points
        }
        try:
            # logger.info(f"question: {item['question']}")
            content_embeddings = generate_embeddings(item['question'])
            # logger.info(f"Embeddings: {content_embeddings}")
            item['contentVector'] = content_embeddings
        except Exception as e:
            # logger.error(f"Error in generate_embeddings function: {e}",exc_info=True)
            print(f"Error in generate_embeddings function: {e}",exc_info=True)

        # item['titleVector'] = title_embeddings
        item['@search.action'] = 'upload'
        # logger.info(f"Embeddings created and inserted in cosmos DB for query {id}")

    except Exception as e:
        print(f"An error occurred: {e}", exc_info=True)
        # logger.error(f"An error occurred: {e}", exc_info=True)

    temp = [item]
    try:
        qa_client.upload_documents(temp)
        # logger.info(f"Uploaded item {id}")
    except Exception as e:
        print(f"Error uploading batch {id}", exc_info=True)
        # logger.info(f"Error uploading batch {id}", exc_info=True)


#---------------- QnA Similarity---------------------------

def get_similar_qa(query, exclude_category):
    # try:
        qa_client = SearchClient(endpoint=service_endpoint, index_name=qa_index_name, credential=azure_search_credential)
        k=1
        print(query)
        num_doc = qa_client.get_document_count()

        if int(num_doc) > 0:
            ## SEARCH SCORE
            results = qa_client.search(
                search_text= None,
                vector=generate_query_embeddings(query),
                top_k=5,
                vector_fields="contentVector",
                filter = f'exclude_category eq {str(exclude_category)}',
                # vector=Vector(value=generate_query_embeddings(query), k=5, fields="contentVector"),
                select=[ 'id',
                    'question',
                    'answer',
                    'exclude_category',
                    'thoughts',
                    'data_points'
                    ],
                top=1,
                
            ) 
            
            print("RERSULT=================")
            # print(results)
            result = list(results)
            # logger.info(result[0]['question'])
            # logger.info(result[0]['@search.score'])
            # logger.info(f"{str(result[0]['question'])} - {str(result[0]['@search.score'])}")
            similarity_sensitivity = 0.95
            # logger.info(f"similarity_sensitivity - {str(similarity_sensitivity)}")
            if len(result) > 0:
                set1 = set(ast.literal_eval(result[0]['exclude_category']))
                set2 = set(exclude_category)

                intersection = set1.intersection(set2)
                if len(intersection) == len(set1) == len(set2):
                    if float(result[0]['@search.score']) > similarity_sensitivity:
                        return result
        return None