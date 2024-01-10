import logging, json, os, requests, datetime, base64
import azure.functions as func
import openai
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Configuring logging options
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# OpenAI Parameters
openai.api_type = "azure"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv('OPENAI_DEPLOYMENT_NAME', "text-davinci-003")
TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', 0.7))
MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', 800))
TOP_P = float(os.getenv('OPENAI_TOP_P', 0.95))
FREQUENCY_PENALTY = float(os.getenv('OPENAI_FREQUENCY_PENALTY', 0.0))
PRESENCE_PENALTY = float(os.getenv('OPENAI_PRESENCE_PENALTY', 0.0))
PROMPT = os.getenv('OPENAI_PROMPT', "You are a JSON formatter for extracting information out of a single chat conversation.\n\nSummarize the conversation, key: summary\nIs the customer satisfied with the agent interaction (Yes or No), key: satisfied\n\nAnswer in JSON machine-readable format, using the keys from above.\nPretty print the JSON and make sure that it is properly closed at the end and do not generate any other content.")
MODEL_TYPE = os.getenv('OPENAI_MODEL_TYPE', 'text').lower()

# Azure Cognitive Search Parameters
OPENAI_PROMPT_KEYS = os.getenv('OPENAI_PROMPT_KEYS', 'summary')
search_service = os.getenv('AZURE_SEARCH_SERVICE_NAME')
index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
api_key = os.getenv('AZURE_SEARCH_API_KEY')
api_version = os.getenv('AZURE_SEARCH_API_VERSION')

def get_openai_response(text, prompt= ''):
    if MODEL_TYPE == 'chat':
        return get_openai_chat_completion(text, prompt)
    else:
        return get_openai_completion(text, prompt)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_openai_completion(text, prompt = ''):
    prompt = PROMPT if prompt == '' else prompt
    response = openai.Completion.create(
              engine=MODEL,
              prompt=f"{text}:\n\n{prompt}",
              temperature=TEMPERATURE,
              max_tokens=MAX_TOKENS,
              top_p=TOP_P,
              frequency_penalty=FREQUENCY_PENALTY,
              presence_penalty=PRESENCE_PENALTY
            )
    return response['choices'][0]['text']

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_openai_chat_completion(text, prompt = ''):
    prompt = PROMPT if prompt == '' else prompt
    response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages = [
                {"role":"system","content": prompt},
                {"role":"user","content":text}],
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
    return response['choices'][0]['message']['content']
    
def send_to_queue(data, queue_name="openai_results_queue", scheduled_enqueue_time_utc=datetime.datetime.utcnow()):
    servicebus_client = ServiceBusClient.from_connection_string(conn_str=os.getenv('AzureServiceBusConnectionString'), logging_enable=True)
    with servicebus_client:
        sender = servicebus_client.get_queue_sender(queue_name=queue_name)
        with sender:
            sender.send_messages(ServiceBusMessage(json.dumps(data), scheduled_enqueue_time_utc=scheduled_enqueue_time_utc))


def push_to_ACS(data):
    # POST https://[service name].search.windows.net/indexes/[index name]/docs/index?api-version=[api-version]   
    #   Content-Type: application/json   
    #   api-key: [admin key]
    url = f'https://{search_service}.search.windows.net/indexes/{index_name}/docs/index?api-version={api_version}'
    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key
    }
    body = {
        'value': [
        ]
    }
    # Create the body value for the Push Api
    body_value = {
        '@search.action': 'merge',
        'metadata_storage_path': data['metadata_storage_path'],
    }
    if data['output'] == []:
        logging.info(f'No output from OpenAI. Skipping this document {data["metadata_storage_path"]}.')
        return 200

    for key in OPENAI_PROMPT_KEYS.replace(' ','').split(','):
        key, search_type, _ = key.split(':')
        if search_type == 'Edm.String':
            body_value[key] = str(data['output'][key])
        elif search_type == 'Edm.Int32':
            body_value[key] = int(data['output'][key])       
        elif search_type == 'Edm.Int64':
            body_value[key] = int(data['output'][key])
        elif search_type == 'Edm.Double':
            body_value[key] = float(data['output'][key])
        elif search_type == 'Edm.Boolean':
            body_value[key] = bool(data['output'][key])
        else:
            logging.error(f'Invalid search type {search_type} for key {key}.')
            return 500

    body['value'].append(body_value)
    
    # Write on the index using the Push Api
    response = requests.post(url, headers=headers, json=body)
    logging.info(response.text)
    return response.status_code



def compose_response(json_data):
    values = json.loads(json_data)['values']
    
    # Prepare the Output before the loop
    results = {}
    results["values"] = []
    
    for value in values:
        output_record = transform_value(value)
        if output_record != None:
            results["values"].append(output_record)
    return json.dumps(results, ensure_ascii=False)

# Perform an operation on a record
def transform_value(value):
    try:
        recordId = value['recordId']
    except AssertionError  as error:
        return None

    # Validate the inputs
    try:         
        assert ('data' in value), "'data' field is required."
        data = value['data']        
        assert ('metadata_storage_path' in data), "'metadata_storage_path' field is required in 'data' object."
        assert ('text' in data), "'text' field is required in 'data' object."
    except AssertionError  as error:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "Error:" + error.args[0] }   ]       
            })

    try:        
        # Send message on Service Bus Queue
        output = {
            "metadata_storage_path": data['metadata_storage_path'],
            "text": data['text']
        } 
        send_to_queue(output, queue_name="openai_queue")


    except Exception as e:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": f"Could not complete operation for record. {e}" }   ]       
            })

    return ({
            "recordId": recordId,
            "data": {
                "status": "Processing document"
                    }
            })