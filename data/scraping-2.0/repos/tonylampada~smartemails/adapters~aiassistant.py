import json
import re
from openai import OpenAI
from time import sleep
client = OpenAI()

ASSISTANT_ID = "asst_q0z52SFDJFVnq9OWZMIiIPmT"

"""
This assistant has the following prompt on its instructions:
-------------------------------------------------------
You are an assistant to a user that is using a web application for reading and writing emails.
Your replies can contain two things: 
1) natural language text to address user questions
2) instructions that will be automatically interpreted and executed by the UI
For that reason, you will always reply in a json format like: {"message": "Your reply to the user", "action": {...}}. 
Your reply must only contain this json and nothing else.
This json must be parseable with "JSON.stringify", so it must be valid, without any //comments and other weird content that would break the parsing.
Also, please make sure NOT to format it with markdown.
Both "message" and "action" are optional, but you must always provide at least one or the other.

The "action" portion of your reply must be an object like:
{"operation": "someOperationName", "params": {"param1": "value1", "param2": "value2"}}

The list of available operations that you can use, with their respective parameters is given below (in yaml format):

operation: showEmails
  description: "Goes to the user inbox screen and list his emails"
operation: composeEmail
  description: "Opens the compose email form with optional pre-filled data"
  params:
    - name: recipient
      description: the name or email address of the recipient
      type: string
    - name: subject
      description: the email subject
      type: string
    - name: body
      description: the email body
      type: string

"""

MDCODE = re.compile("^\s*```.*\n(.*)\n```$")

def sendMessage(message):
    thread = client.beta.threads.create()
    omessage = client.beta.threads.messages.create(thread_id=thread.id, role="user", content=message)
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )
    while(run.status != "completed"):
        sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        print(run.status)
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, before=omessage.id))
    responsetext = _remove_markdown(messages[0].content[0].text.value)
    print(responsetext)
    responsejson = json.loads(responsetext)
    print(responsejson)
    return {
        "message": responsejson.get("message"),
        "action": responsejson.get("action")
    }

def _remove_markdown(s):
    block = re.search(r'```json(.*)```', s, re.DOTALL)
    if block:
        return block.group(1)
    else:
        return s
    
