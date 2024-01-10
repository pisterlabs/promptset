#   Copyright [2023] [Holosun ApS]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import traceback
import base64
import json
import datetime

from langchain.schema import Document

from ..components import get_embeddings, pick_vectorstore, load_memories
from ..logging import setup_logging

logging = setup_logging()

def embed_pubsub_chunk(data: dict):
    """Triggered from a message on a Cloud Pub/Sub topic "embed_chunk" topic
    Will only attempt to send one chunk to vectorstore.
    Args:
         data JSON
    """

    message_data = base64.b64decode(data['message']['data']).decode('utf-8')
    messageId = data['message'].get('messageId')
    publishTime = data['message'].get('publishTime')

    logging.debug(f"This Function was triggered by messageId {messageId} published at {publishTime}")
    logging.debug(f"from_pubsub_to_supabase message data: {message_data}")

    try:
        the_json = json.loads(message_data)
    except Exception as err:
        logging.error(f"Error - could not parse message_data: {err}: {message_data}")
        return "Could not parse message_data"

    if not isinstance(the_json, dict):
        raise ValueError(f"Could not parse message_data from json to a dict: got {message_data} or type: {type(the_json)}")

    page_content = the_json.get("page_content", None)
    if page_content is None:
        return "No page content"
    if len(page_content) < 100:
        logging.warning(f"too little page content to add: {message_data}")
        return "Too little characters"
    
    metadata = the_json.get("metadata", None)

    vector_name = metadata.get("vector_name", None)
    if vector_name is None:
        msg = f"FATAL: No vector name was found within metadata: {metadata}"
        logging.error(msg)
        return msg
    
    logging.info(f"Embedding: {vector_name} page_content: {page_content[:30]}...")

    if 'eventTime' not in metadata:
        metadata['eventTime'] = datetime.datetime.utcnow().isoformat(timespec='microseconds') + "Z"

    doc = Document(page_content=page_content, metadata=metadata)

    # init embedding and vector store
    embeddings = get_embeddings(vector_name)

    memories = load_memories(vector_name)
    vectorstore_list = []
    for memory in memories:  # Iterate over the list
        for key, value in memory.items(): 
            logging.info(f"Found memory {key}")
            vectorstore = value.get('vectorstore', None)
            if vectorstore is not None:
                vectorstore = pick_vectorstore(vectorstore, vector_name=vector_name, embeddings=embeddings)
                vs_retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))
                vectorstore_list.append(vs_retriever)

    # can have multiple vectorstores per embed
    metadata_list = []
    for vector_store in vectorstore_list:
        logging.debug(f"Adding single document for {vector_name} to vector store {vector_store}")
        try:
            vector_store.add_documents([doc])
            logging.info(f"Added doc for {vector_name} to {vector_store} - metadata: {metadata}")
            metadata_list.append(metadata)
        except Exception as err:
            error_message = traceback.format_exc()
            logging.error(f"Could not add document {doc} for {vector_name} to {vector_store} - metadata: {metadata}: {str(err)} traceback: {error_message}")

    return metadata_list
