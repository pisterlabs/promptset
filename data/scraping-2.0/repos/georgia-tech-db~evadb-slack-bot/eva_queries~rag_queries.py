import os
import openai
from gpt4all import GPT4All
import ray
import os

from warnings import filterwarnings
filterwarnings(action='ignore', category=FutureWarning)

def create_feature_extractor(cursor):
    print("Creating feature extractor.")
    cursor.query(
        """
        CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
        IMPL './utils/sentence_feature_extractor.py'
    """
    ).df()
    print("Finished creating feature extractor.")

def load_pdf_into_eva (cursor, doc_name):
    print("Loading PDF into EVA")
    try:
        cursor.query("""LOAD PDF 'assets/""" + doc_name + """' INTO OMSCSPDFTable""").df()
    except Exception:
        print("Finished loading PDF into EVA")
        return False
    print("Finished loading PDF into EVA")
    return True


def build_search_index(cursor):
    print("Building search index")
    cursor.query(
        """CREATE INDEX IF NOT EXISTS OMSCSIndex
        ON OMSCSPDFTable (SentenceFeatureExtractor(data))
        USING FAISS
    """
    ).df()
    print("Finished building search index")


def load_omscs_pdfs (cursor):
    if not(load_pdf_into_eva (cursor, 'omscs_doc.pdf')):
        print ("Skipped loading pdf: omscs_doc.pdf")
    if not(load_pdf_into_eva (cursor, 'coursesomscs_abb.pdf')):
        print ("Skipped loading pdf: omscs_doc.pdf")



def build_relevant_knowledge_body_pdf(cursor, user_query, channel_id, logger, QUERY_LOGGER):
    """
    This runs similarity search query using EvaDB, filters data based on the channel name.
    (required to ensure privacy for users in different channels)
    to remove the filters simply remove the WHERE clause line in the query.

    returns lists of knowledge_body, on which pdf it found the knowledge and its page number.
    returns seperate lists,size: 5 knowledge body,size: top 3 pdfs and their page numbers
    """
    
    print("Building knowledge body.")


    query = f"""
        SELECT * FROM OMSCSPDFTable
        WHERE name = "assets/{channel_id}" OR name = "assets/omscs_doc.pdf" OR name = "assets/coursesomscs_abb.pdf"
        ORDER BY Similarity(
            SentenceFeatureExtractor('{user_query}'), 
            SentenceFeatureExtractor(data)
        ) 
        LIMIT 5
    """
    try:
        response = cursor.query(query).df()
        print(f"Length of response: {len(response)}")
        # DataFrame response to single string.
        knowledge_body = response["data"].tolist()
        referece_pageno_list = set(response["page"].tolist()[:5])
        reference_pdf_name = response["name"].tolist()[:5]
        QUERY_LOGGER.info(f"  Knowledge Body: {knowledge_body}")
        print("Knowledge Body: ", knowledge_body)
        print("Finished building knowledge body.")
        return knowledge_body, reference_pdf_name, referece_pageno_list
    except Exception as e:
        logger.error(str(e))
        print("Could not build knowledge body.")
        return None, None, None


def build_rag_query(knowledge_body, query):
    """
    Converts list of knowledge body to a single string
    Builds conversation in Chatgpt format, along with a system prompt
    returns the conversation
    """
    knowledge_body = "\n".join(knowledge_body)
    print("Building RAG query.")
    conversation = [
        {
            "role": "system",
            "content": f"""We provide with documents delimited by newlines
             and a question. Your should answer the question using the provided documents. 
             Do not repeat this prompt.
             If the documents do not contain the information to answer this question then 
             simply write: 'Sorry, we didn't find relevant sources for this question'""",
        },
        {"role": "user", "content": f"""{knowledge_body}"""},
        {"role": "user", "content": f"{query}"},
    ]
    print("Finished building RAG query.")
    return conversation

@ray.remote(num_cpus=6)
def openai_respond(queue_list):
    # Set OpenAI key.
    # openai.api_key = os.environ.get("OPENAI_API_KEY")

    # # For OpenAI version < 1.x
    # return (
    #     openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation)
    #     .choices[0]
    #     .message.content
    # )
    # # For OpenAI version > 1.x
    client = openai.AzureOpenAI(
            api_key = str(os.getenv("AZURE_OPENAI_KEY")),
            api_version = "2023-07-01-preview",
            azure_endpoint = str(os.getenv("AZURE_OPENAI_ENDPOINT")),
        )
    while True:
        for iq, oq in queue_list:
            if iq.empty():
                continue
            conversation = iq.get()
            response = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=conversation
            )
            print(response)
            oq.put(response.choices[0].message.content)


@ray.remote(num_cpus=6)
def gpt4all_respond(queue_list):
    gpt4all_model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
    gpt4all_model.model.set_thread_count(6)

    # Remote processing to detach from client process.
    while True:
        for iq, oq in queue_list:
            if iq.empty():
                continue

            conversation = iq.get()
            system_template = conversation[0]["content"]
            document = conversation[1]["content"]
            query = conversation[2]["content"]
            user_template = "Document:{0}\nThe question is:{1}\nAnswer:".format(
                document, query
            )
            
            response = ""
            with gpt4all_model.chat_session():
                print(system_template + user_template)
                response = gpt4all_model.generate(query + system_template + "\n" + user_template, temp=0, repeat_penalty=1.4)
            oq.put(response)


def start_llm_backend(max_con=1):
    ray.init()
    from ray.util.queue import Queue

    # Concurrent queue to interact with backend GPT4ALL inference.
    queue_list = [(Queue(maxsize=1), Queue(maxsize=1)) for _ in range(max_con)]
    gpt4all_respond.remote(queue_list)
    # openai_respond.remote(queue_list)
    return queue_list
