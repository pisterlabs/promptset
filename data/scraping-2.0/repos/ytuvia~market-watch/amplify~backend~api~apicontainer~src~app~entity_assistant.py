from openai import OpenAI
from lib.appsync import query_api
from pdf_utils import crop_pdf
from textract_utils import list_document_images, list_document_titles
from vision_utils import image_question

import json
import os
import time
import boto3
import traceback

s3 = boto3.client('s3')

def get_secret(secret_name):
    region_name = os.environ.get('REGION');

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    return secret

api_key = get_secret('OPENAI_API_KEY')
GPT_MODEL="gpt-4-1106-preview"
client = OpenAI(
    api_key=api_key 
)

def run_assistance_thread(entity_id, thread_id, message):
    entity = get_entity(entity_id)

    assistant = client.beta.assistants.retrieve(
        assistant_id=entity['assistant']['id']
    )
    thread = client.beta.threads.retrieve(
        thread_id=thread_id
    )
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message,
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    run = wait_on_run(run, thread)

    return run

def run_assistance(entity_id, message):
    (assistant, thread) = build_infrastructure(entity_id)
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message,
    )
    show_json(message)

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    show_json(run)
    run = wait_on_run(run, thread)
    show_json(run)

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    show_json(messages)
    return messages

def build_infrastructure(entity_id, newThread=False):
    entity = get_entity(entity_id)
    saved_documents = entity['documents']['items']
    saved_assistant = entity['assistant']
    saved_threads = entity['threads']['items']

    if(saved_assistant):
        assistant = client.beta.assistants.retrieve(
            assistant_id=saved_assistant.get('id'),
        )
    else:
        instructions = assistant_instructions(entity_id)
        image_extract = image_extract_function()
        list_images = list_images_function()
        list_titles = list_titles_function()
        image_question = image_question_function()
        assistant = client.beta.assistants.create(
            name=entity.get('name'),
            instructions=instructions,
            tools=[
                {"type": "retrieval"},
                image_extract,
                list_images,
                image_question
            ],
            model=GPT_MODEL,
        )
        create_entity_assistant(entity, assistant)
    assistant = emmbed_assistant_documents(assistant, saved_documents)

    show_json(assistant)
    if(len(saved_threads)>0 and not newThread):
        saved_threads = sorted(saved_threads, key=lambda x: x["updatedAt"], reverse=True)
        saved_thread = saved_threads[0]
        thread = client.beta.threads.retrieve(
            thread_id=saved_thread.get('id')
        )
    else:
        thread = client.beta.threads.create()
        create_entity_thread(entity, thread)
    
    show_json(thread)

    return (assistant, thread)

def emmbed_assistant_documents(assistant, documents):
    #file_ids = [obj['id'] for obj in documents]
    file_ids = [item['id'] for item in documents if item.get("filename", "").lower().endswith(".pdf")]
    try:
        assistant = client.beta.assistants.update(
            assistant.id,
            file_ids=file_ids,
        )
    except Exception as e:
        print(e)

    show_json(assistant)
    return assistant

def download_file(key):
  filename = os.path.basename(key)
  tmp_file_path = f'/tmp/{filename}'
  s3.download_file(os.environ.get('STORAGE_CONTENT_BUCKETNAME'), key, tmp_file_path)
  return tmp_file_path

def get_entity(id):
    variables = {
        'id': id,
    }
    query = """
        query GetEntity($id: ID!) {
            getEntity(id: $id) {
                id
                name
                documents {
                    items{
                        id
                        filename
                    }
                }
                assistant {
                    id
                }
                threads {
                    items {
                        id,
                        status,
                        updatedAt
                    }
                }
            }
        }
    """
    response = query_api(query, variables)
    data = response['data']['getEntity']
    return data

def get_thread(id):
    variables = {
        'id': id,
    }
    query = """
        query GetEntityThread($id: ID!) {
            getEntityThread(id: $id) {
                id,
                status,
                entityThreadsId,
                updatedAt
            }
        }
    """
    response = query_api(query, variables)
    data = response['data']['getEntityThread']
    return data

def get_thread_messages(thread_id):
    thread = get_thread(thread_id)
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    return (thread_id, thread['status'], thread['entityThreadsId'], thread['updatedAt'], messages.data)

def create_entity_assistant(entity, assistant):
    variables = {
        'input':{
            'id': assistant.id,
            'name': assistant.name
        }
    }
    query = """
        mutation createEntityAssistant($input: CreateEntityAssistantInput!) {
            createEntityAssistant(input: $input) {
                id
            }
        }
    """
    response = query_api(query, variables)

    variables = {
        'input':{
            'id': entity['id'],
            'entityAssistantId': assistant.id
        }
    }
    query = """
        mutation UpdateEntity($input: UpdateEntityInput!) {
            updateEntity(input: $input) {
                id
            }
        }
    """
    response = query_api(query, variables)
    return assistant

def create_entity_thread(entity, thread):
    variables = {
        'input':{
            'id': thread.id,
            'entityThreadsId': entity.get('id')
        }
    }
    query = """
        mutation createEntityThread($input: CreateEntityThreadInput!) {
            createEntityThread(input: $input) {
                id
            }
        }
    """
    response = query_api(query, variables)
    return thread

def get_entity_thread(entity_id):
    entity = get_entity(entity_id)
    saved_threads = entity.get('threads').get('items')
    if len(saved_threads) > 0:
        saved_thread = saved_threads[0]
        thread = client.beta.threads.retrieve(
            thread_id=saved_thread.get('id')
        )
        status = saved_thread['status']
        return (thread, status)
    else:
       (assistant, thread) = build_infrastructure(entity_id)
       return (thread, 'completed')

def get_entity_messages(entity_id):
    (thread, status) = get_entity_thread(entity_id)
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    show_json(messages)
    return (thread.id, status, messages.data)

def update_document_file(id, openai_id):
    variables = {
        'input':{
            'id': id,
            'openaiFileId': openai_id
        }
    }
    query = """
        mutation UpdateDocument($input: updateDocumentInput!) {
            updateDocument(input: $input) {
                id
            }
        }
    """
    response = query_api(query, variables)
    return response['data']['updateDocument']

def show_json(obj):
    #print(json.loads(obj.model_dump_json()))
    return

def wait_on_run(run, thread):
    current = run.status
    while run.status == "queued" or run.status == "in_progress" or run.status == "requires_action":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        print(run.status, flush=True)
        if run.status == "requires_action":
            tools = run.required_action.submit_tool_outputs.tool_calls
            for tool in tools:
                tool_call_id = tool.id
                if tool.type == "function":
                    try:
                        result = execute_tool(tool)
                        print(result, flush=True)
                        run = client.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread.id,
                            run_id=run.id,
                            tool_outputs=[
                                {
                                    "tool_call_id": tool_call_id,
                                    "output": result,
                                },
                            ]
                        )
                    except Exception as e:
                        print(e)
                        run = client.beta.threads.runs.cancel(
                            thread_id=thread.id,
                            run_id=run.id,
                        )
                        

        if current != run.status:
            current = run.status
            update_status(thread, current)
                        
        time.sleep(5)
    return run

def update_status(thread, status):
    variables = {
        'input':{
            'id': thread.id,
            'status': status
        }
    }
    query = """
        mutation UpdateEntityThread($input: UpdateEntityThreadInput!) {
            updateEntityThread
            (input: $input) {
                id
                status
                title
                createdAt
                updatedAt
                entityThreadsId
            }
        }
    """
    response = query_api(query, variables)
    return response['data']['updateEntityThread']

def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

def delete_thread(entity_id):
    (thread, status) = get_entity_thread(entity_id)
    variables = {
        'input':{
            'id': thread.id
        }
    }
    query = """
        mutation DeleteEntityThread($input: DeleteEntityThreadInput!) {
            deleteEntityThread(input: $input) {
                id
            }
        }
    """
    response = query_api(query, variables)

    thread = client.beta.threads.delete(
        thread.id
    )
    return thread

def delete_entity_embedding(entity_id):
    entity = get_entity(entity_id)
    saved_documents = entity['documents']['items']
    for document in saved_documents:
        file = client.files.delete(
            document.get('id')
        )

def delete_assistant(entity_id):
    entity = get_entity(entity_id)
    assistant = entity.get('assistant', None)
    if assistant:
        variables = {
            'input':{
                'id': assistant.get('id')
            }
        }
        query = """
            mutation DeleteEntityAssistant($input: DeleteEntityAssistantInput!) {
                deleteEntityAssistant(input: $input) {
                    id
                }
            }
        """
        response = query_api(query, variables)

        assistant = client.beta.assistants.delete(
            entity['assistant']['id']
        )
    
    return assistant

def delete_file(file_id):
    variables = {
        'input':{
            'id': file_id
        }
    }
    query = """
        mutation DeleteDocument($input: DeleteDocumentInput!) {
            deleteDocument(input: $input) {
                id
                entityDocumentsId
            }
        }
    """
    response = query_api(query, variables)

    file = client.files.delete(
        file_id
    )

    return response['data']['deleteDocument']

def archive_thread(entity_id, thread_id):
    update_status({'id': thread_id}, 'archived')
    result = build_infrastructure(entity_id, True)
    return result

def assistant_instructions(entity_id):
    instructions = """you are an assistant for a book publisher researcher. the reasearcher needs help with analyzing a book content and structure.
    if there is requirement to extract images from the boook, list the book images first. """
    instructions = instructions + f" use the following configured properties: configured_entity_id={entity_id}"
    return instructions

def image_extract_function():
    return {
            "type": "function",
            "function": {
                "name": "image_extract",
                "description": "Extract the image from the document",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "configured_entity_id": {
                            "type" : "string",
                            "description" : "the id of the entity the assistant is configured to"
                        },
                        "page": {
                            "type" : "integer",
                            "description" : "the image location page number"
                        },
                        "bounding_box": {
                            "type": "object", 
                            "description": "top, left, width and height of image bounding box",
                            "properties": {
                                "top": {
                                    "type": "number"
                                },
                                "left": {
                                    "type": "number"
                                },
                                "width": {
                                    "type": "number"
                                },
                                "height": {
                                    "type": "number"
                                }
                            }
                        },
                        "polygon": {
                            "type": "array", 
                            "description": "image polygon points",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {
                                        "type": "number"
                                    },
                                    "y": {
                                        "type": "number"
                                    }
                                }
                            }
                        },
                    },
                    "required": ["configured_entity_id", "page", "polygon","bounding_box"]
                }
            }
        }

def list_images_function():
    return {
        "type": "function",
        "function": {
            "name": "list_images",
            "description": "list the images in the document",
            "parameters": {
                "type": "object",
                "properties": {
                    "configured_entity_id": {
                        "type" : "string",
                        "description" : "the id of the entity the assistant is configured to"
                    },
                },
                "required": ["configured_entity_id"]
            }
        }
    }

def list_titles_function():
    return {
        "type": "function",
        "function": {
            "name": "list_titles",
            "description": "list the titles in the document",
            "parameters": {
                "type": "object",
                "properties": {
                    "configured_entity_id": {
                        "type" : "string",
                        "description" : "the id of the entity the assistant is configured to"
                    },
                },
                "required": ["configured_entity_id"]
            }
        }
    }

def image_question_function():
    return {
        "type": "function",
        "function": {
            "name": "image_question",
            "description": "ask a question about an image",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type" : "string",
                        "description" : "the url of the image"
                    },
                    "question": {
                        "type" : "string",
                        "description" : "the question about the image"
                    },
                },
                "required": ["url", "question"]
            }
        }
    }

def execute_tool(tool):
    func = tool.function
    id = tool.id
    name = func.name
    if name == "image_extract":
        arguments = json.loads(func.arguments)
        entity = get_entity(arguments.get('configured_entity_id'))
        saved_documents = entity['documents']['items']
        pdf_documents = [item for item in saved_documents if item.get("filename", "").lower().endswith(".pdf")]
        if len(pdf_documents) > 0:
            document = pdf_documents[0]
            filename = document.get('filename')
            result = crop_pdf(os.environ.get('STORAGE_CONTENT_BUCKETNAME'), filename, arguments.get('page'), arguments.get('bounding_box'))
            return result
    if name == "list_images":
        arguments = json.loads(func.arguments)
        entity = get_entity(arguments.get('configured_entity_id'))
        saved_documents = entity['documents']['items']
        json_documents = [item for item in saved_documents if item.get("filename", "").lower().endswith(".json")]
        if len(json_documents) > 0:
            document = json_documents[0]
            filename = document.get('filename')
            result = list_document_images(os.environ.get('STORAGE_CONTENT_BUCKETNAME'), filename)
            return json.dumps(result)
    if name == "list_titles":
        arguments = json.loads(func.arguments)
        entity = get_entity(arguments.get('configured_entity_id'))
        saved_documents = entity['documents']['items']
        json_documents = [item for item in saved_documents if item.get("filename", "").lower().endswith(".json")]
        if len(json_documents) > 0:
            document = json_documents[0]
            filename = document.get('filename')
            result = list_document_titles(os.environ.get('STORAGE_CONTENT_BUCKETNAME'), filename)
            return json.dumps(result)
    if name == "image_question":
        arguments = json.loads(func.arguments)
        url = arguments.get('url')
        question = arguments.get('question')
        result = image_question(url, question)
        return result
    
    raise ValueError("there is no implementation for the selected function")
