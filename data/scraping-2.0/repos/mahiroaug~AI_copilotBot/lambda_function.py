### lambda_function.py
from __future__ import print_function

import json
import re
import os
from datetime import datetime, timedelta

import boto3
from botocore.exceptions import ClientError

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import openai


### function list ####################################
functions = [
{
    "name": "get_checklist",
    "description": "Get the checking items and hints for each service systems",
    "parameters": {
        "type": "object",
        "properties": {
            "service_system": {
                "type": "string",
                "description": "target system, 以下のいずれかのみ指定可能。「認証基盤」,「WEBサーバ」,「ネットワーク機器」"
            },
        },
        "required": ["service_system"]
    },
},
]

### function ####################################
def get_checklist(service_system):
    file_mapping = {
        "認証基盤": "aaa.json",
        "WEBサーバ": "web.json",
        "ネットワーク機器": "network.json"
    }
    print("get_checklist.file_mapping: ",file_mapping.get(service_system))
    
    file_path = "checklists/" + file_mapping.get(service_system)
    print("get_checklist.file_path: ",file_path)
    
    with open(file_path, 'r') as file:
        jsonData = file.read()
    return json.loads(jsonData)




### lambda ####################################
def lambda_handler(event, context):
    print("event: ", event)
    
    # prevent dual launch
    if "X-Slack-Retry-Num" in event["headers"]:
        return {"statusCode": 200, "body": json.dumps({"message": "No need to resend"})}
    
    
    ### initializer ####################################
    
    # Get secrets from Secrets-Manager
    secret_dict = json.loads(get_secret())
    slack_client = WebClient(secret_dict["SLACK_OAUTH_TOKEN"])
    openai.organization = secret_dict["OPENAI_ORGANIZATION"]
    openai.api_key = secret_dict["OPENAI_API_KEY"]
    
    body = json.loads(event["body"])
    text = re.sub(r"<@.*>", "", body["event"]["text"])
    channel = body["event"]["channel"]
    thread_ts = body["event"].get("thread_ts") or body["event"]["ts"]
    userId = body["event"]["user"]
    print("input: ", text, "channel: ", channel, "thread:", thread_ts)
    
    
    
    ### preparation ####################################
    
    # get thread messages
    thread_messages_response = slack_client.conversations_replies(channel=channel, ts=thread_ts)
    messages = thread_messages_response["messages"]
    messages.sort(key=lambda x: float(x["ts"]))
    #print("messages:",messages)
    
    # get recent 30 messages in the thread
    prev_messages = [
        {
            "role": "assistant" if "bot_id" in m and m["bot_id"] else "user",
            "content": re.sub(r"<@.*>|`info: prompt.*USD\)` ", "", m["text"]),
        }
        for m in messages[0:][-30:]
    ]
    print("prev_messages:",prev_messages)
    
    
    
    ### COMPLETION ######################################
    
    # make responce with system_prompt from base-model
    with open(os.environ["ENV_SYSTEM_PROMPT_BASE"], 'r') as file:
        system_prompt = file.read()
    
    # set model
    model=os.environ["ENV_GPT_MODEL"]
    prompt=[
        {
            "role": "system",
            "content": system_prompt
        },
        *prev_messages
    ]
    print("mdoel:",model,"prompt:",prompt)
    
    # STEP1 : preCompletion
    print("STEP1")
    response1 = Cpmpletion_function_auto(model,prompt)
    message1 = response1["choices"][0]["message"]
    print("message1: ",message1)
    #post_slack(slack_client, channel, json.dumps(message1), thread_ts)
    message1content = message1["content"]
    print("message1.content: ",message1content)
    if message1content:
        post_slack(slack_client, channel, message1content, thread_ts)
    
    # STEP2 : exec my function
    if message1.get("function_call"):
        print("STEP2")
        function_name = message1["function_call"]["name"]
        arguments = json.loads(message1["function_call"]["arguments"])
        function_response = exec_my_function(function_name,arguments)
        
        # STEP3 : Completion
        print("STEP3")
        prompt2=[
            *prompt,
            message1,
            {
                "role": "function",
                "name": function_name,
                "content": json.dumps(function_response),
            },
        ]
        print("mdoel:",model,"prompt2:",prompt2)
        response2 = Completion_assistant(model,prompt2)
        make_assistans_message_and_post_slack(response2, slack_client, channel, thread_ts)
    
    return {"statusCode": 200}

def Cpmpletion_function_auto(model,prompt):
    try:
        # step1 : Completion(function_call)
        print("step1 - Completion(function_call)")
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            functions=functions,
            function_call="auto",
        )
        print("response1: " + json.dumps(response))
        return response
    except Exception as err:
        print("Error: ", err)

def exec_my_function(function_name,arguments):
    print("exec_my_function: function_name=",function_name," arguments=",arguments)
    print("arguments.get: ",arguments.get("service_system"))
    if function_name == "get_checklist":
        function_response = get_checklist(
            service_system=arguments.get("service_system")
        )
        print("function_response: ",function_response)
        return function_response
    else:
        print("no match function")
        return 0

def Completion_assistant(model,prompt):
    # step3 : completion
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt
        )    
        print("response2: " + json.dumps(response))
        return response
    except Exception as err:
        print("Error: ", err)

def make_assistans_message_and_post_slack(openai_response, slack_client, channel, thread_ts):
    
    # calculate tokens
    tkn_pro = openai_response["usage"]["prompt_tokens"]
    tkn_com = openai_response["usage"]["completion_tokens"]
    tkn_tot = openai_response["usage"]["total_tokens"]
    cost = tkn_tot * 0.002 / 1000
    msg_head = "\n `info: prompt + completion = %s + %s = %s tokens(%.4f USD)` " % (tkn_pro,tkn_com,tkn_tot,cost)
    res_text = openai_response["choices"][0]["message"]["content"]
    answer = res_text + msg_head
    print("answer:",answer)
    
    # post_message
    post_slack(slack_client, channel, answer, thread_ts)
    
    return res_text

def post_slack(slack_client, channel, text, thread_ts):
    try:
        response = slack_client.chat_postMessage(
            channel=channel,
            text=text,
            as_user=True,
            thread_ts=thread_ts,
            reply_broadcast=False
        )
        print("slackResponse: ", response)
    except SlackApiError as e:
        print("Error posting message: {}".format(e))
        

def get_secret():
    secret_name = os.environ["ENV_SECRET_NAME"]
    region_name = os.environ["ENV_REGION_NAME"]
    
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e
    
    #####print("secret dir",get_secret_value_response['SecretString'])
    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    
    # Your code goes here.
    return secret