import json
import os
import sys
import logging
import boto3
import botocore
import asyncio
import aiohttp
from redisDB.config import Redis
from redisDB.cache import Cache
from agent.chatAgent import ChatAgent
from schema.chat import Message
from langchain.schema import (
    AIMessage,
    HumanMessage,
    # SystemMessage
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

dynamodb = boto3.resource('dynamodb')
connections = dynamodb.Table(os.environ['TABLE_NAME'])
redis = Redis()

async def handler(event, context):
    logger.debug("sendmessage: %s" % event)

    redis_client = await redis.create_connection()

    userMessage = json.loads(event.get('body', '{}')).get('data')
    domain_name = event.get('requestContext',{}).get('domainName')
    stage       = event.get('requestContext',{}).get('stage')
    connectionId = event.get('requestContext',{}).get('connectionId')
    token = connections.get_item(Key={'connectionId': connectionId})['Item']['token']

    if (userMessage and domain_name and stage) is None:
        return { 'statusCode': 400, 
                 'body': 'bad request' }

    answer = await askChatAgent(userMessage, token)

    result = sendMessage(answer, connectionId, token, domain_name, stage)
    return result
    

async def askChatAgent(chat, token):
    print("Beginning ask chat agent function")
    chatAgent = ChatAgent()
    json_client = redis.create_rejson_connection()
    cache = Cache(json_client)

    # Create a new message instance and add to cache, specifying the source as human
    msg = Message(sender="human",msg=chat)

    await cache.add_message_to_cache(token=token, source="human", message_data=msg.dict())

    # Get chat history from cache
    data = await cache.get_chat_history(token=token)

    # reconstruct chat to langchain memory format
    # print(data)
    # get last 4 messages
    messages = data['messages'][-4:]

    chat_history = [HumanMessage(content=message['msg']) if message['sender'] == "human" else AIMessage(content=message['msg']) for message in messages]
    # print(chat_history)

    chatMessage = {
        "id": token,
        "message": msg,
        "history": chat_history
    }
    print(chatMessage["history"])
    chatAgent.updateChatHistory(chat_history=chatMessage["history"])

    try:
        res = chatAgent.query(input=chatMessage["message"].msg)
        res = res['output']
    except Exception as e:
        res = str(e)
        if res.startswith("Could not parse LLM output: `"):
            # need to run python 3.9+ to use removeprefix!
            res = res.removeprefix("Could not parse LLM output: `").removesuffix("`")

    print(res)
    msg = Message(
        sender="bot",
        msg=res
    )

    print(msg)

    stream_data = {}
    stream_data[str(token)] = json.dumps(msg.dict())
    await cache.add_message_to_cache(token=token, source="bot", message_data=msg.dict())

    return res

def sendMessage(answer, connectionId, token,domain_name, stage,):
    apigw_management = boto3.client('apigatewaymanagementapi',
                                    endpoint_url=F"https://{domain_name}/{stage}")

    try:
        _ = apigw_management.post_to_connection(ConnectionId=connectionId,
                                                Data=answer)
    except botocore.exceptions.ClientError as e:
        if e.response.get('ResponseMetadata',{}).get('HTTPStatusCode') == 410:
            connections.delete_item(Key={'connectionId': connectionId})
            logger.debug('post_to_connection skipped: %s removed from connections' % connectionId)
        else:
            logger.debug('post_to_connection failed: %s' % e)
            return { 'statusCode': 500,
                     'body': 'something went wrong' }

    return { 'statusCode': 200,
             'body': 'ok' }

def main(event, context):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(handler(event, context))