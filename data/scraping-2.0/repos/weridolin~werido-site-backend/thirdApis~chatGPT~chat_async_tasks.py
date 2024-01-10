import openai
from ws.const import ChatGPTMessageType
from asgiref.sync import sync_to_async
from thirdApis.models import ChatGPTMessage
import json
import uuid
import openai
import os

@sync_to_async
def create_record(query_content, conversation_id, parent_message_uuid,role=None,children_message_uuid=None):
    return ChatGPTMessage.objects.create(
        content=query_content,
        conversation_id=conversation_id,
        parent_message_uuid=parent_message_uuid,
        children_message_uuid=children_message_uuid,
        role=role or 0,
        content_type="text"
    )


async def chatGPT_create_request(
        data=None,
        ws=None):
    try:
        if isinstance(data,str):
            data = json.loads(data)
        query_content, conversation_id, msg_id, parent_message_uuid = data.get("query_content"), data.get("conversation_id"),\
            data.get("msg_id"), data.get("parent_message_uuid")

        
        query_message = await create_record(query_content=query_content, conversation_id=conversation_id, parent_message_uuid=parent_message_uuid)
        openai.api_key=os.environ["OPENAPI_SECRET"]
        prompt = "\nHuman:{content}.\nAI:".format(
            content=query_content)  # todo 上下文
        response = await openai.Completion.acreate(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=2046,
            temperature=0.9,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
            stop=[" Human:", " AI:"]
        )
        reply_message_uuid = str(uuid.uuid4())
        # reply_message = await create_record(
        #     query_content=response["choices"][0]['text'],
        #     conversation_id=conversation_id,
        #     parent_message_id=query_message.id,
        #     role=1
        # )
        print(">>>>>>>>>>>",response["choices"][0]['text'])
        reply = {
            "type": ChatGPTMessageType.reply,
            "data": {
                "query_content": "",
                "parent_message_uuid": query_message.uuid,
                "reply_content": response["choices"][0]['text'],
                "uuid": reply_message_uuid,
                "conversation_id": conversation_id,
            },
        }
        await ws.send(text_data=json.dumps(reply,ensure_ascii=False))
        return query_message,reply.get("data")

    except Exception as exc:
        reply = {
            "type": ChatGPTMessageType.error,
            "data": {
            },
        }
        await ws.send(text_data=json.dumps(reply,ensure_ascii=False))


def chatGPT_callback(future):
    ## 更新下此次查询和回复的上下文信息
    query_message,reply_data = future.result()
    query_message.children_message_uuid=reply_data["uuid"]
    query_message.save()
    ## 更新回答到数据库
    ChatGPTMessage.objects.create(
        content=reply_data["reply_content"],
        conversation_id=reply_data["conversation_id"],
        parent_message_uuid=query_message.uuid,
        uuid = reply_data["uuid"],
        children_message_uuid= None,
        role=1,
        content_type="text"
    )