
import json
from bson import ObjectId
from langchain.schema import (
  AIMessage,
  HumanMessage
)

import src.user.service as user_service
import src.openai_module.service as openai_service
from src.databases.mongo import db
from .models import ChatModel, MessageModel
from src.user.models import UserModel
from src.whatsapp_webhook.schemas.webhook_payload_schema import (
  WaMessage,
  WaContact,
  WaPayloadMetadata
)
from src.databases.vector import vector_db
from src.utils.dict import flatten_dict
from src.openai_module.prompts import (
  system_general_prompt,
  system_introduction_prompt,
)

async def get_or_create_chat(chat_criteria: dict, n_messages: int) -> ChatModel:
  try:
    chat = await get_chat(chat_criteria, n_messages)

    if not chat:
      chat = await create_chat(ChatModel(**chat_criteria))
      
    return chat
  except Exception as e:
    raise e  

async def create_chat(chat_dto: ChatModel) -> ChatModel:
  try:
    inserted = await db["chats"].insert_one({
      "_id": chat_dto.id,
      "user": chat_dto.user.dict(),
      "system_profile": chat_dto.system_profile.dict(),
      "messages": []
    })

    chat = await get_chat({"_id": inserted.inserted_id}, n_messages=10)

    return chat
  except Exception as e:
    print("Error at create_chat", e)
    
    raise e

async def get_chat(chat_data: dict, n_messages: int) -> ChatModel | None:
  try:
    if chat_data.get("_id", None):
      chat_data["_id"] = ObjectId(chat_data["_id"])
    
    chat_data = flatten_dict(chat_data)

    pipeline = [
      {"$match": chat_data},
      {"$unwind": {"path": "$messages", "preserveNullAndEmptyArrays": True}},
      {"$sort": {"messages.created_at": -1}},
      {"$limit": n_messages},
      {"$group": {
        "_id": "$_id",
        "system_profile": { "$first": "$system_profile" },
        "user": { "$first": "$user" },
        "messages": {"$push": "$messages"}
      }}
    ]

    chat = await db["chats"].aggregate(pipeline).to_list(length=1)

    if chat:
      return ChatModel(**(chat[0]))

    return None
  except Exception as e:
    print("Error at get_chat", e)
    
    raise e

async def upsert_chat_message(chat_data: dict, messages: list[MessageModel]) -> ChatModel | None:
  updated = await db["chats"].update_one(
    chat_data,
    {
      "$push": {
        "messages": {
          "$each": [{
            "_id": message.id,
            "wa_message_id": message.wa_message_id,
            "content": message.content,
            "role": message.role,
            "created_at": message.created_at,
          } for message in messages]
        }
      }
    },
    upsert=True
  )

  chat = await db["chats"].find_one({"_id": updated.upserted_id})

  if not chat:
    return None

  return ChatModel(**chat)

async def handle_chat_message_pipeline(
  message: WaMessage, 
  contact: WaContact,
  metadata: WaPayloadMetadata
):
  try:
    # * Get/Create user
    user = await user_service.get_or_create_user(
      UserModel(
        whatsapp_id=contact.wa_id,
        name=contact.profile.name,
        phone_number=contact.wa_id
      )
    )

    # * Get/Create chat
    user_chat = await get_or_create_chat({
      "user":  {
        "whatsapp_id": user.whatsapp_id,
      },
      "system_profile": {
        "whatsapp_id": metadata.phone_number_id
      }
    }, n_messages=10);
  
    # * Format chat history into langchain chat prompt
    chat_history = format_chat_history(user_chat, user)
    new_message = HumanMessage(content=message.text.body)
    
    # * Intent classification
    chat_completion: AIMessage = handle_intent_spec_message(
      chat_history=chat_history,
      new_message=new_message,
      user=user
    )

    return chat_completion
    
  except Exception as e:
    print("Error at handle msg pipeline", e)
    
    raise e
  
def handle_intent_spec_message(
  chat_history: list[AIMessage | HumanMessage],
  new_message: AIMessage | HumanMessage, 
  user: UserModel
) -> AIMessage:
  prompt = system_general_prompt
  
  if len(chat_history) == 0:
    prompt = system_introduction_prompt

  vector_res = vector_db.client.query.get(
    "Topics",
    ["topic"]
  ).with_near_text({
    "concepts": f"{new_message.content}"
  }).with_limit(3).do()

  print(json.dumps(vector_res, indent=2))

  chat_completion: AIMessage = openai_service.create_full_chat_completion(
    message_history=[*chat_history, new_message],
    prompt=prompt,
    additional_data=user.bio_information.dict() if user.bio_information else {
      "full_name": "",
      "date_of_birth": "",
      "gender": ""
    }
  )

  return chat_completion

def format_chat_history(chat: ChatModel, user: UserModel) -> list[AIMessage | HumanMessage]:
  formatted_chat_history = []

  for message in chat.messages:
    if message.role == "assistant":
      formatted_chat_history.append(
        AIMessage(content=message.content)
      )

      continue

    if message.role == "user":
      formatted_chat_history.append(
        HumanMessage(content=message.content)
      )

  return formatted_chat_history