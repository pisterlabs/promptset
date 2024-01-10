################
## The Agent ###
################

import json
import boto3
import os
import time

from db_utils import query,save_item_ddb,update_item_session,update_items_out
from agent_utils import langchain_agent

from langchain.agents import load_tools,Tool
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from utils import (whats_reply)
from langchain.retrievers import AmazonKendraRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory

client_s3 = boto3.client('s3')
dynamodb_resource=boto3.resource('dynamodb')
bedrock_client = boto3.client("bedrock-runtime")
kendra_client=boto3.client(service_name='kendra')

table_name_active_connections = os.environ.get('whatsapp_MetaData')
table_session_active = dynamodb_resource.Table(os.environ['TABLE_SESSION_ACTIVE'])
kendra_index_id = os.environ.get('KENDRA_INDEX')

key_name_active_connections = os.environ.get('ENV_KEY_NAME')
Index_Name = os.environ.get('ENV_INDEX_NAME')

base_path="/tmp/"

table_name_session = os.environ.get('TABLE_SESSION')
lambda_query_function_name = os.environ.get('ENV_LAMBDA_QUERY_NAME')
model_id = os.environ.get('ENV_MODEL_ID')
whatsapp_out_lambda = os.environ.get('WHATSAPP_OUT')

table = dynamodb_resource.Table(table_name_active_connections)

model_parameter = {"temperature": 0.0, "top_p": .9, "max_tokens_to_sample": 350}
llm = Bedrock(model_id=model_id, model_kwargs=model_parameter,client=bedrock_client)

def memory_dynamodb(id,table_name_session,llm):
    message_history = DynamoDBChatMessageHistory(table_name=table_name_session, session_id=id)
    memory = ConversationBufferMemory(
        memory_key="chat_history", llm=llm,max_token_limit=800,chat_memory=message_history, return_messages=True,ai_prefix="A",human_prefix="H"
    )
    return memory
    
    
def promp_definition():

    prompt_template = """
        You are an assistant in the airline La Inventada who answers to users factual information about the status of the passengers through their ID or rservation number, deliver information to help the passenger and also do casual conversation. 

        Use the following format:
        History: the context of a previous conversation with the user. Useful if you need to recall past conversation, make a summary, or rephrase the answers. if History is empty it continues.
        Question: the input question you must answer
        Thought: you should always think about what to do, also try to follow steps mentioned above.Idenfity if user wants to do casual chat, search passanger information or search knowledge base. Also try to follow steps mentioned above. You must undestand the identification ID as number, no words.
        Action: the action to take, should be one of ["search-passanger-information","search_knowledge_base"] or use your sympathy.
        Action Input: the input to the action
        Observation: the result of the action
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question, always reply in the original user language and human legible.

        History: 
        {chat_history}

        Question: {input}

        Assistant:
        {agent_scratchpad}"""

    updated_prompt = PromptTemplate(
    input_variables=['chat_history','input', 'agent_scratchpad'], template=prompt_template)

    return updated_prompt


def kendra_tool(llm,kendra_index_id):
    retriever = AmazonKendraRetriever(index_id=kendra_index_id)
    memory_kendra = ConversationBufferMemory(memory_key="chat_history", return_messages=True,ai_prefix="A",human_prefix="H")

    Kendra_prompt_template = """Human: 
    The following is a friendly conversation between a human and an AI. 
    The AI is an assistant in the airline La Inventada and deliver information to help the passengers and provides specific details with but limits it to 240 tokens.
    If the AI does not know the answer to a question, it truthfully says it does not know.

    Assistant: OK, got it, I'll be a talkative truthful assistant.

    Human: Here are a few documents in <documents> tags:
    <documents>
    {context}
    </documents>
    Based on the above documents, provide a detailed answer for, {question} 
    Answer "don't know" if not present in the document. 

    Assistant:
    """
    PROMPT = PromptTemplate(
        template=Kendra_prompt_template, input_variables=["context","question"]
    )

    condense_qa_template_kendra = """{chat_history}
    Human:
    Given the following conversation and a follow up question, rephrase the follow up question 
    to be a standalone question.

    Standalone Question:

    Assistant:"""

    standalone_question_prompt_kendra = PromptTemplate.from_template(condense_qa_template_kendra)

    qa_kendra = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever, 
            condense_question_prompt=standalone_question_prompt_kendra, 
            return_source_documents=False, 
            combine_docs_chain_kwargs={"prompt":PROMPT},
            memory = memory_kendra,
            #verbose=True
            )
    return qa_kendra


def lambda_handler(event, context):
    print (event)

    whats_message = event['whats_message']
    print(whats_message)
    whats_token = event['whats_token']
    messages_id = event['messages_id']
    phone = event['phone']
    phone_id = event['phone_id']
    phone_number = phone.replace("+","")

    #The session ID is created to store the history of the chat. 

    try:
        session_data = query("phone_number",table_session_active,phone_number)
        now = int(time.time())
        diferencia = now - session_data["session_time"]
        if diferencia > 300:  #session time in seg
            update_item_session(table_session_active,phone_number,now)
            id = str(phone_number) + "_" + str(now)
        else:
            id = str(phone_number) + "_" + str(session_data["session_time"])

    except:
        now = int(time.time())
        new_row = {"phone_number": phone_number, "session_time":now}
        save_item_ddb(table_session_active,new_row)
        
        id = str(phone_number) + "_" + str(now)

    try:
        print('REQUEST RECEIVED:', event)
        print('REQUEST CONTEXT:', context)
        print("PROMPT: ",whats_message)

        #s = re.sub(r'[^a-zA-Z0-9]', '', query)

        tools = load_tools(
                        ["awslambda"],
                        awslambda_tool_name="search-passanger-information",
                        awslambda_tool_description="useful for searching passenger data by their ID, only send the number",
                        function_name=lambda_query_function_name,
                    )
        
        qa_kendra = kendra_tool(llm,kendra_index_id)
        
        tools.append(
            Tool.from_function(
                func=qa_kendra.run,
                name="search_knowledge_base",
                description="Searches and returns documents regarding to help the passenger",
            )
            )

        memory = memory_dynamodb(id,table_name_session,llm)

        agent = langchain_agent(memory,tools,llm)

        agent.agent.llm_chain.prompt=promp_definition()
        response = agent(whats_message)
        print(response)

        whats_reply(whatsapp_out_lambda,phone, whats_token, phone_id, f"{response['output']}", messages_id)
        

        end = int(time.time())

        update_items_out(table,messages_id,response['output'],end)
                
        return({"body":response['output']})
        
        
    except Exception as error: 
            print('FAILED!', error)
            return({"body": "Cuek! I dont know"})

