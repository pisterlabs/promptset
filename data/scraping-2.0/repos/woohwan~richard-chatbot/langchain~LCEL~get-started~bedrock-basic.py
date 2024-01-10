from langchain.prompts import PromptTemplate  
from langchain_core.output_parsers import StrOutputParser

# create boto3 session
import boto3
session = boto3.session.Session(profile_name='default')

# create bedrock client from session
bedrock_client = session.client(
  service_name='bedrock-runtime',
  region_name='us-east-1')

# create bedrock chat model
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

model = Bedrock(
  client=bedrock_client,
  model_id='anthropic.claude-v2',
  model_kwargs={ 'max_tokens_to_sample': 512 },
  streaming=True,
  callbacks=[StreamingStdOutCallbackHandler()],
)

prompt = PromptTemplate.from_template("tell me a short joke about {topic}")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "ice cream"})