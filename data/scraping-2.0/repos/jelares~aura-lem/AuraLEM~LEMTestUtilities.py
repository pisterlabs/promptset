import simplejson as json
from openai import OpenAI
import os
from DynamoDBUtilities import *
from AuraELAM.UDSUtilities import validate_response

def analyze_async(event, context):
  # Get the chat history from step function input
  api_key = str(event["api_key"])
  uid = str(event["uid"])
  iid = str(event["iid"])
  elam_response_mtl = int(event["elam_response_mtl"])
  batch = tuple(event["limits"])

  print("\n\nRunning analyze_async with batch: ", batch, "\n\n")

  message_history_partitionk = api_key + uid + iid + 'messages'
  message_history = full_limit_query(message_history_partitionk, False, int(batch[0]))

  # Remove any elements beyond the batch[1]th element in the array message_history
  message_history = message_history[batch[1]:]
  print("message history: ", message_history, "\n\n")

  # Get API key from environment
  openai_key = os.environ['openai_key']
  openai_client = OpenAI(api_key=openai_key)
  
  # Specify the AI model
  model = "gpt-4-1106-preview"
  
  # create the UDS parition key
  UDS_partition_key = api_key + uid + iid + 'UDS'
  latest_UDS = full_limit_query(UDS_partition_key, False)

  # Check if latest_UDS exists, if not create a blank one
  if latest_UDS == []:
    latest_UDS = {
      'basic_info': {
        'name': '',
        'current_location': '',
        'occupation': '',
        'sex': ''
      },
      'traits': [],
      'skills': [],
      'factual_history': [],
      'summary': ''
    }

  else:
    latest_UDS = latest_UDS[0]['uds']

  system_message = f"""
    You are being used via API in an LLM chat app which can 'learn' its own users through its chats with them. It works through AI agents: you are such an agent, located in our backend. Your output will never be directly seen by the user but rather you are a reasoning module whose purpose is to initialize & modify the user understanding data structure (UDS) based off chat history.
    
    UDS: a data-dense, JSON, plaintext format for storing information about personalities. Spec below:
    {{
    'basic_info': {{
    'name': 'Jesus Lares',
    'current_location': 'San Francisco',
    'occupation': 'Startup founder',
    'sex': 'male'
    }}
    'traits': [
    ['trait name', (int) % strength of trait from 0-100, 'few words on reasoning & evidence.'],
    ],
    'skills': [
    ['skill name', (int) % strength of trait from 0-100, 'few words on reasoning & evidence.'],
    ],
    'factual_history' : [
    'history event name'
    ]
    'summary': 'a 1-2 paragraph (depending on how much information you have) textual summary encompassing a wholistic view of the users' personality, history, and achievements.'
    }}

    --- Further explanation on UDS ---
    We want the UDS to be short and precise so that only the most prominent & revelaing user info is presented. Unless you think an entry allows you to infer something revealing, you need not add it. For the traits, skills, and factual_history arrays, limit it to the 10 most revealing entries each (max) and remove, combine, or modify them however you feel is best to capture more information about the user.

    Examples:
    traits entry: ['bold', 80, "loves risk, exemplified by his decision to found a startup, rock climb, and go on month-long backpacking trips."]
    skills entry: ['physics', 60, "undergraduate physics degree from MIT."]
    factual_history entry: "did his undergraduate at MIT completing physics and computer science degrees."

    --- Final notes ---
    You are a backend AI. Your entire purpose is to create or modify a UDS. You will be given a (potentially empty) UDS at the end of this system message. You must ONLY respond with a modified UDS following the key name spec above and nothing more. This is extremely important, as any extraneous input will cause the system to crash.

    --- Current User UDS Below ---
    {json.dumps(latest_UDS)}
  """

  user_message = "What can you infer about the user from the following chat history? Again please respond ONLY with a modified UDS: \n\n"
  
  # note that chat history goes from most recent to oldest, so must be reversed
  i = len(message_history) - 1
  while i >= 0:
    message = message_history[i]
    user_message += f"{message['role']}: {message['content']}\n\n"

    i -= 1

  messages = [
    {
      "role": "system",
      "content": system_message
    },
    {
      "role": "user",
      "content": user_message
    }
  ]
  
  response = openai_client.chat.completions.create(model=model,
  messages=messages,
  stop=None,
  response_format={"type": "json_object"},
  max_tokens=elam_response_mtl)


  validated, modified_UDS = validate_response(response)

  # If validation fails try again once more with new message
  if not validated:
    print("VALIDATION FAILED. NEEDED RETRY.")

    messages.append({
      "role": "assistant",
      "content": response
    })

    messages.append({
      "role": "user",
      "content": "This response is not a valid UDS, incorrect syntax. Please try one more time. It is EXTREMELY IMPORTANT that you get the syntax correct as per the system message."
    })
      
    response = openai_client.chat.completions.create(model=model,
    messages=messages,
    stop=None,
    response_format={"type": "json_object"},
    max_tokens=elam_response_mtl)

    validated, modified_UDS = validate_response(response)
    
  # If validation fails again, return error
  if not validated:
    return {
      'statusCode': 500,
      'body': json.dumps('Failed to generate UDS.')
    }

  # at this point, after a max of one retry, it is necessarily valid. Else would have failed.

  # create the new UDS sort key
  print("modified UDS: ", modified_UDS, "\n\n")

  item={
    'partitionk': UDS_partition_key,
    'sortk': get_sortk_timestamp(),
    'uds': modified_UDS
  }

  uds_put_response = put_item_ddb(item)

  if uds_put_response['statusCode'] == 200:
    return {
      'statusCode': 200,
      'body': json.dumps('Successfully updated UDS.')
    }
  
  else:
    return {
      'statusCode': 500,
      'body': json.dumps('Failed to update UDS.')
    }


"""
Fake class mimicing the AWS API Gateway connection object for ws
"""
class FakeConn:
  def __init__(self):
    pass

  def post_to_connection(self, **kwargs):
    # Print the new characters without a newline and flush the output
    if json.loads(kwargs['Data'])['status'] != "complete":
      print(json.loads(kwargs['Data'])["message"], end='', flush=True)
    else:
      print("[COMPLETE]")

class FakeLambdaClient:
  def __init__(self):
    pass

  def invoke(self, **kwargs):
    analyze_async(json.loads(kwargs["Payload"]), {})