'''Create a conversationPipeline
Embedding: openai
DB: postgres, with label based filtering
framework: langchain
'''

import os
import glob
import sys
 
# setting path
sys.path.append('../app')

from core.pipeline import ConversationPipeline, DataUploadPipeline
from core.embedding.openai import OpenAIEmbedding
from core.vectordb.postgres4langchain import Postgres
import schema

######## Configure the pipeline's tech stack ############
RESOURCE_LABEL = "ESV-Bible"
embedding_tech = OpenAIEmbedding()
vectordb_tech = Postgres(host="localhost", port='5435',
    collection_name="aDotBCollection_fromTSV".lower(),
    user="postgres",
    password="password",
    embedding=embedding_tech,
    label=RESOURCE_LABEL)
chat_stack = ConversationPipeline(
    embedding=embedding_tech,
    vectordb=vectordb_tech,
    user="test_user")
chat_stack.set_llm_framework(schema.LLMFrameworkType.LANGCHAIN,
    api_key=os.getenv('OPENAI_API_KEY'),
    model='gpt-3.5-turbo',
    vectordb=chat_stack.vectordb)
chat_stack.label = RESOURCE_LABEL

##### Checking DB has content ##########

# print(chat_stack.vectordb.db_conn.get(
#     include=["metadatas"]
# ))


########### Talk #################

QUERY = "Who created the earth?"
bot_response = chat_stack.llm_framework.generate_text(
                query=QUERY, chat_history=chat_stack.chat_history)
print(f"Human: {QUERY}\nBot:{bot_response['answer']}\n"+\
    f"Sources:{[item.metadata['source'] for item in bot_response['source_documents']]}\n\n")
chat_stack.chat_history.append((bot_response['question'], bot_response['answer']))

QUERY = "What happended then?"
bot_response = chat_stack.llm_framework.generate_text(
                query=QUERY, chat_history=chat_stack.chat_history)
print(f"Human: {QUERY}\nBot:{bot_response['answer']}\n"+\
    f"Sources:{[item.metadata['source'] for item in bot_response['source_documents']]}\n\n")
chat_stack.chat_history.append((bot_response['question'], bot_response['answer']))


QUERY = "What is light? Is it sun?"
bot_response = chat_stack.llm_framework.generate_text(
                query=QUERY, chat_history=chat_stack.chat_history)
print(f"Human: {QUERY}\nBot:{bot_response['answer']}\n"+\
    f"Sources:{[item.metadata['source'] for item in bot_response['source_documents']]}\n\n")
chat_stack.chat_history.append((bot_response['question'], bot_response['answer']))

print("!!!!!!!!!!!!!! Finished !!!!!!!!!!!!!!!!")
