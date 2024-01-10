from typing import Final

from boto3.session import Session
from common import settings
from langchain.chains import LLMChain
from langchain.llms import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.prompts import PromptTemplate

SESSION_ID: Final = "1"


class BedrockService:
    MODEL_ID = "anthropic.claude-v2"
    MODEL_KWARGS = {"temperature": 0.0, "max_tokens_to_sample": 1000}
    REGION_NAME = "us-east-1"
    TABLE_NAME = "SessionTable"

    def __init__(self) -> None:
        self.session = Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=self.REGION_NAME,
        )
        bedrock_runtime = self.session.client("bedrock-runtime")
        self.llm = Bedrock(
            client=bedrock_runtime,
            model_id=self.MODEL_ID,
            model_kwargs=self.MODEL_KWARGS,
        )
        message_history = DynamoDBChatMessageHistory(
            table_name=self.TABLE_NAME,
            session_id=SESSION_ID,
            boto3_session=self.session,
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=message_history, return_messages=True
        )

    def call_endpoint(self, prompt: PromptTemplate, query: str) -> str:
        llm_chain = LLMChain(
            llm=self.llm, prompt=prompt, verbose=False, memory=self.memory
        )
        result = llm_chain.predict(Query=query)
        return result
