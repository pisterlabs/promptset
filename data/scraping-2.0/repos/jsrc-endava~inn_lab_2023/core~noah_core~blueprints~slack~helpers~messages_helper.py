import os
from noah_core.models.slack_message import SlackMessage
from noah_core.extensions import db
from langchain.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings

def process_message(message_id, channel_id, user_id, team_id, username, text):
  message = save_message(message_id, channel_id, user_id, team_id, username, text)
  # TODO: Modify when source becomes variable
  vector_id = save_message_vector(message.id, text, 'slack')
  if vector_id:
    message = update_message(message, vector_id)

  return message

def save_message(message_id, channel_id, user_id, team_id, username, text):
  print("Saving message")
  message = SlackMessage(
    message_id=message_id, channel_id=channel_id, user_id=user_id, team_id=team_id, username=username, text=text
  )
  db.session.add(message)
  db.session.commit()
  db.session.refresh(message)
  return message

def update_message(message, vector_id):
  message.vector_id = vector_id
  db.session.commit()
  db.session.refresh(message)
  return message

def save_message_vector(message_id, text, source):
  # TODO: Using this vector store in similar way on interactions blueprint, maybe move to a helper?
  vector_store = Milvus(
    embedding_function=OpenAIEmbeddings(),
    collection_name="messages",
    connection_args={
      "uri": os.environ['ZILLIZ_ENDPOINT'],
      "token": os.environ['ZILLIZ_TOKEN'],
    }
  )

  ids = vector_store.add_texts(texts=[text], metadatas=[{'message_id': message_id, 'source': source}])
  print(ids)
  return ids[0] if ids else None
