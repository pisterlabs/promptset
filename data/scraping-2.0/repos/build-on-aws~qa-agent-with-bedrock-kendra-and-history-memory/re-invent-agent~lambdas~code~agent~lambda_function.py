import boto3
import os
from langchain.llms.bedrock import Bedrock
from langchain.retrievers import AmazonKendraRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools,Tool
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.agents import initialize_agent, AgentType

dynamodb_resource=boto3.resource('dynamodb')
boto3_bedrock = boto3.client("bedrock-runtime")
kendra_client=boto3.client(service_name='kendra')

#table_name_active_connections = os.environ.get('TABLE_CONNECTIONS')

table_name = os.environ.get('TABLE_SESSION')
table_name_agenda = os.environ.get('TABLE_NAME')
kendra_index_id = os.environ.get('KENDRA_INDEX')
lambda_query_function_name = os.environ.get('LAMBDA_QUERY_NAME')
bedrock_model_id = os.environ.get('MODEL_ID')
model_parameter = {"temperature": 0.0, "top_p": .5, "max_tokens_to_sample": 2000}

llm = Bedrock(model_id=bedrock_model_id, model_kwargs=model_parameter,client=boto3_bedrock)

def kendra_retriever(kendra_index_id,llm):
    retriever = AmazonKendraRetriever(index_id=kendra_index_id)
    memory_kendra = ConversationBufferMemory(memory_key="chat_history", return_messages=True,ai_prefix="A",human_prefix="H")


    Kendra_prompt_template = """Human: You are an Agenda re:Invent 2023 Assistant. 
    You are talkative and provides specific details from its context.
    If the AI does not know the answer to a question, it truthfully says it 
    does not know.

    Assistant: OK, got it, I'll be a talkative truthful AI assistant.

    Human: Here are a few documents in <documents> tags:
    <documents>
    {context}
    </documents>
    Based on previous documents, provide a list of data for each documents (If you find more than one), example: spaker veneu, date, thirdpartyid... , it's okay if you respond with just one option, the information is important so I can decide the best choise for me to: {question}

    Assistant:
    """
    PROMPT = PromptTemplate(
        template=Kendra_prompt_template, input_variables=["context", "question"]
    )

    condense_qa_template_kendra = """{chat_history}
    Human:
    Given the previous conversation and a follow up question below, based on previous documents, provide a list of data for each document (If you find more than one), example: spaker veneu, date, thirdpartyid... , it's okay if you respond with just one option, the information is important so I can decide the best choise for me.
    Follow Up Question: {question}
    Standalone Question:

    Assistant:"""
    standalone_question_prompt_kendra = PromptTemplate.from_template(condense_qa_template_kendra)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        condense_question_prompt=standalone_question_prompt_kendra, 
        return_source_documents=False, 
        combine_docs_chain_kwargs={"prompt":PROMPT},
        memory = memory_kendra,
        #verbose=True
        )

    return qa

def define_kendra_tools(qa,tools):
    tools.append(
    Tool.from_function(
        func=qa.run,
        name="re-invent-agenda-2023",
        description="useful when you want search sessions in re:Invent 2023 agenda. This will output documentation in text type, and /n, you must deliver a coherent and accurate response from all the documentation provided. ",
    )
    )
    return tools

def promp_definition():

    prompt_template = """
        You are an assistant who provides recommendations on re:invent 2023 sessions, you deliver the recommendation according to their question, and also do casual conversation. 
        Use the following format:
        History: the context of a previous conversation with the user. Useful if you need to recall past conversation, make a summary, or rephrase the answers. if History is empty it continues.
        Question: the input question you must answer
        Thought: you should always think about what to do, Also try to follow steps mentioned above.
        Action: the action to take, should be one of ['re-invent-agenda-2023',"search-session-id-information"], provides options that have the most information that can be useful in making a decision.
        Action Input: the input to the action
        Observation: the result of the action
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question, If you have more than one answer, give them all. Always reply in the original user language and human legible.

        History: 
        {chat_history}

        Question: {input}

        Assistant:
        {agent_scratchpad}"""

    updated_prompt = PromptTemplate(
    input_variables=['chat_history','input', 'agent_scratchpad'], template=prompt_template)

    return updated_prompt

def memory_dynamodb(id,table_name_session,llm):
    message_history = DynamoDBChatMessageHistory(table_name=table_name_session, session_id=id)
    memory = ConversationBufferMemory(
        memory_key="chat_history", llm=llm,max_token_limit=800,chat_memory=message_history, return_messages=True,ai_prefix="A",human_prefix="H"
    )
    return memory

def langchain_agent(memory,tools,llm):
    zero_shot_agent=initialize_agent(
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    #verbose=True,
    max_iteration=1,
    #return_intermediate_steps=True,
    handle_parsing_errors=True,
    memory=memory
)
    return zero_shot_agent

def lambda_handler(event, context):
    print (event)
    prompt = event['prompt']
    session_id = event['session_id']

    tools = load_tools(
                        ["awslambda"],
                        awslambda_tool_name="search-session-id-information",
                        awslambda_tool_description="useful for searching session information data by their ID number, only send the ID number",
                        function_name=lambda_query_function_name,
                    )

    qa = kendra_retriever(kendra_index_id,llm)
    tools = define_kendra_tools(qa,tools)
    memory  = memory_dynamodb(session_id,table_name,llm)
    agent = langchain_agent(memory,tools,llm)
    agent.agent.llm_chain.prompt=promp_definition()
    completion = agent(prompt)

    print(completion['output'])
    return completion['output']
