from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations import ConversationAnalysisClient
import azure.cosmos.cosmos_client as cosmos_client
import azure.functions as func
from openai import OpenAI
import logging
import os
import json
import time
from prompt import ASSISTANT_PROMPT, CONVERSATION_SUMMARY_PROMPT

OPENAI_API_KEY = os.environ['OPENAI_KEY']
COSMOSDB_URI = os.environ['COSMOSDB_URI']
COSMOSDB_KEY = os.environ['COSMOSDB_KEY']
COSMOSDB_DATABASE_ID = os.environ['COSMOSDB_DATABASE_ID']
COSMOSDB_CONTAINER_ID = os.environ['COSMOSDB_CONTAINER_ID']
LANGUAGE_KEY = os.environ['LANGUAGE_KEY']
LANGUAGE_ENDPOINT = os.environ['LANGUAGE_ENDPOINT']

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
client = cosmos_client.CosmosClient(COSMOSDB_URI, credential=COSMOSDB_KEY)
database = client.get_database_client(COSMOSDB_DATABASE_ID)
container = database.get_container_client(COSMOSDB_CONTAINER_ID)

oai_client = OpenAI(
   api_key=OPENAI_API_KEY,
)

@app.route(route="create_activity")
def create_activity(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()
    goal = req_body.get('goal')
    activity_context = req_body.get('activity_context', 'No context provided.')

    # Create OpenAI assistant for activity
    logging.info('Creating assistant prompt...')
    prompt = ASSISTANT_PROMPT.format(goal=goal, activity_context=activity_context)
    logging.info(f'Assistant prompt: {prompt}')

    logging.info('Creating assistant...')
    assistant = oai_client.beta.assistants.create(
        name="SIMBA",
        instructions=prompt,
        tools=[{"type": "retrieval"}],
        model="gpt-4-1106-preview"
    )
    logging.info('Success.')
    activity_id = assistant.id
    logging.info(f'Created assistant with id = {activity_id}')

    # Save activity (assistant) ID on CosmosDB
    activity_data = {
        'id': activity_id,
        'goal': goal,
        'prompt': prompt,
        'threads': []
    }

    logging.info(f'Attempting to save activity data to CosmosDB...')
    container.create_item(activity_data)
    logging.info(f'Success.')

    # Create output json with activity ID
    response_dict = {
        'activity_id': activity_id
    }
    return func.HttpResponse(
             json.dumps(response_dict),
             status_code=200
        )


@app.route(route="check_activity")
def check_activity(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()
    activity_id = req_body.get('activity_id')
    data = container.read_item(item=activity_id, partition_key=activity_id)
    threads = data['threads']
    goal = data['goal']

    # Load threads
    if len(threads) == 0:
        logging.info('No threads found.')
        response_dict = {
            'num_conversations': 0,
            'message': 'No conversations found.'
        }
        return func.HttpResponse(
                json.dumps(response_dict),
                status_code=200
            )
    
    logging.info(f'Loading threads: {threads}')

    conversations = []

    for thread_id in threads:
        # Get messages in thread
        messages = oai_client.beta.threads.messages.list(thread_id = thread_id)
        
        # Create conversartion items
        conversation_items = [
            {
                "text": msg.content[0].text.value, 
                "id": msg.id,
                "role": msg.role,
                "participantId": f'STUDENT_{thread_id}' if msg.role == 'user' else 'SIMBA'
            } for msg in messages]
        conversation_items = conversation_items[::-1]
        
        # Create conversation data dict
        conversation_data = {
            "conversationItems": conversation_items,
            "modality": "text",
            "id": thread_id,
            "language": "en",
        }
        
        # Append conversation
        conversations.append(conversation_data)

    tasks = [
        {
            "taskName":"narrativeTask",
            "kind":"ConversationalSummarizationTask",
            "parameters":{"summaryAspects": ["narrative"]}
        }
    ]
    
    # Summarize conversations Azure Language Analytics
    lang_client = ConversationAnalysisClient(LANGUAGE_ENDPOINT, AzureKeyCredential(LANGUAGE_KEY))
    with lang_client:
        poller = lang_client.begin_conversation_analysis({
            "displayName": "Summarize conversations",
            "analysisInput": {
                "conversations": conversations
            },
            "tasks": tasks
        })
        result = poller.result()

    # Give overall summary using OpenAI
    logging.info('Creating OpenAI prompt...')
    conversations_analysis = result['tasks']['items'][0]['results']['conversations']
    conversation_summaries = [res['summaries'][0]['text'] for res in conversations_analysis]
    conversations_summary_prompt = CONVERSATION_SUMMARY_PROMPT.format(conversation_summaries=conversation_summaries)
    response = oai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": conversations_summary_prompt},
            {"role": "user", "content": "Provide a summary. Focus on the students, not on SIMBA. Give one ore two short, actionable advices to the teacher on how they could proceed based on this information."},
        ]
     )
    overall_summary = response.choices[0].message.content

    # Create output json with overall summary
    response_dict = {
        'num_conversations': len(threads),
        'message': overall_summary
    }

    return func.HttpResponse(
            json.dumps(response_dict),
            status_code=200
        )


@app.route(route="create_student_conversation")
def create_student_conversation(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    req_body = req.get_json()
    activity_id = req_body.get('activity_id')

    # Create thread on OpenAI
    logging.info('Creating thread...')
    thread = oai_client.beta.threads.create()
    logging.info(f'Created thread with id = {thread.id}')

    # Append thread ID to activity on CosmosDB
    logging.info(f'Attempting to append thread ID to activity on CosmosDB...')
    data = container.read_item(item=activity_id, partition_key=activity_id)
    logging.info(f'read item: {data}')

    # TO-DO: FIX THIS
    if 'threads' in data:
        data['threads'].append(thread.id)
    else:
        data['threads'] = [thread.id]

    logging.info(f'new item: {data}')
    
    container.upsert_item(data)

    # Create output json with thread ID
    response_dict = {
        'thread_id': thread.id
    }

    return func.HttpResponse(
            json.dumps(response_dict),
            status_code=200
        )


@app.route(route="handle_student_message")
def handle_student_message(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    req_body = req.get_json()
    activity_id = req_body.get('activity_id')
    thread_id = req_body.get('thread_id')
    message = req_body.get('message')

    logging.info('Creating message in thread...')
    message = oai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message
    )
    logging.info('Starting run...')
    run = oai_client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=activity_id
    )

    logging.info('Checking run status...')
    run = oai_client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run.id
    )
    
    while run.status not in ["completed", "failed", "cancelled", "expired"]:
        run = oai_client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        time.sleep(1)

    logging.info('Run completed.')

    messages = oai_client.beta.threads.messages.list(
        thread_id=thread_id
    )
    last_message = messages.data[0].content[0].text.value
    logging.info(f'Last message: {last_message}')

    response_dict = {
        'message': last_message
    }   

    return func.HttpResponse(
            json.dumps(response_dict),
            status_code=200
        )