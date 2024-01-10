from langchain.llms import Bedrock

import boto3

session = boto3.Session()

llm = Bedrock(model_id="anthropic.claude-v2")

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory




conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)

conversation.predict(input="Hi there!")
