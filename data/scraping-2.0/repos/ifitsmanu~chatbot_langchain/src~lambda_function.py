# import all the required modules
import boto3
import json
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
# from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.memory import (
    ConversationSummaryBufferMemory,
    # ConversationBufferMemory,
    CombinedMemory,
    ConversationBufferWindowMemory,
)
from langchain.prompts import PromptTemplate


def create_dynamotable(dynamodb,databasename):
    
    try:
        dynamodb.create_table(
            TableName=databasename,
            KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
    except Exception as e:
        # print(e)
        print("{} already exists".format(databasename))


def lambda_handler(event, context):
    session = boto3.Session()
    print("Lambda Function Invoked")
    body = json.loads(event.get("body", "{}"))
    question = body.get("question")
    session_id = body.get("session_id")
    num_recent_conversations=body.get("num_recent_conversations")
    temparature=body.get("temparature")
    summary_token_limit=body.get("summary_token_limit")
    if question is None:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No question was provided"}),
        }
    dynamodb = session.resource("dynamodb", region_name="us-east-1")
    # create_dynamotable(dynamodb,"SessionTable")
    # create_dynamotable(dynamodb,"SessionSummaryTable")
    session_table = dynamodb.Table("SessionTable")
    summary_table = dynamodb.Table("SessionSummaryTable")
    session_table.meta.client.get_waiter("table_exists").wait(TableName="SessionTable")
    summary_table.meta.client.get_waiter("table_exists").wait(
        TableName="SessionSummaryTable"
    )
    llm = OpenAI(temperature=temparature)
    # create a memory object that pulls latest num_recent_conversations from the dynamodb table

    message_history = DynamoDBChatMessageHistory(
        table_name="SessionTable", session_id=session_id
    )

    conv_memory = ConversationBufferWindowMemory(
        chat_memory=message_history,
        k=num_recent_conversations,
        input_key="input",
        memory_key="chat_history_lines",
    )
    # create another memory object that is initialized with the existing summary of the conversation

    response = summary_table.get_item(Key={"SessionId": session_id})

    if response and "Item" in response:
        summary = response["Item"]["Summary"]
    else:
        summary = ""
    summary_memory = ConversationSummaryBufferMemory(
        llm=OpenAI(temperature=temparature),
        memory_key="summary",
        moving_summary_buffer=summary,
        max_token_limit=summary_token_limit,
        input_key="input",
    )
    # create a combined memory object that combines the above two memories
    memory = CombinedMemory(memories=[conv_memory, summary_memory])
    _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Summary of conversation:
    {summary}
    Current conversation:
    {chat_history_lines}
    Human: {input}
    AI:"""

    prompt = PromptTemplate(
        input_variables=["summary", "chat_history_lines", "input"],
        template=_DEFAULT_TEMPLATE,
    )

    # create a conversation chain object

    conversation = ConversationChain(
        llm=llm, verbose=False, memory=memory, prompt=prompt
    )
    answer = conversation.predict(input=question)
    # update the summary table with the latest summary
    summary_table.put_item(
        Item={
            "SessionId": session_id,
            "Summary": summary_memory.load_memory_variables({})["summary"],
        }
    )
    return {"statusCode": 200, "body": json.dumps(answer)}