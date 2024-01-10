import json
import os
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from script.kendra_receiver.kendra_index_retriever import KendraIndexRetriever

region = os.environ["AWS_REGION"]
kendra_index_id = os.environ["KENDRA_INDEX_ID"]
endpoint_name = os.environ["CHATGLM_ENDPOINT"]

def lambda_handler(event, context):
    
    intent_name = event['sessionState']['intent']['name']
    # Dispatch to your bot's intent handlers
    if intent_name == 'FallbackIntent':
        query = event["inputTranscript"]
    
    retriever = KendraIndexRetriever(kendraindex=kendra_index_id, 
        awsregion=region, 
        return_source_documents=False)
    docs = retriever.get_relevant_documents(query)
    if len(docs) == 0:
        answer = "很抱歉，您的问题超出了我回答的范围"
    else:
        chatbot = build_chain(retriever)
        result = run_chain(chatbot, query)
        answer = result['answer']

    message =  {
            'contentType': 'PlainText',
            'content': answer
        }
    session_attributes = get_session_attributes(event)
    fulfillment_state = "Fulfilled"
    response = close(event, session_attributes, fulfillment_state, message)
    return response


def build_chain(retriever):

    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            input_str = json.dumps({"ask": prompt})
            print(input_str)
            return input_str.encode('utf-8')
        
        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read())
            return response_json["answer"]

    content_handler = ContentHandler()

    llm=SagemakerEndpoint(
            endpoint_name=endpoint_name, 
            region_name=region, 
            model_kwargs={"temperature":0.03, "max_length": 500},
            content_handler=content_handler
        )

    prompt_template = """
    下面是一段人与 AI 的友好对话。 
    AI 很健谈，并根据其上下文提供了许多具体细节。
    如果 AI 不知道问题的答案，它会如实说出不知道。
    说明：请根据 {context} 中的内容，用中文为 {question} 提供详细的答案，字数在50字以内。
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=False
    )
    return qa

def run_chain(chain, prompt: str):
    result = chain(prompt)
    # To make it compatible with chat samples
    return {
        "answer": result['result']
    }
    
def close(event, session_attributes, fulfillment_state, message):
    event['sessionState']['intent']['state'] = fulfillment_state
    return {
        'sessionState': {
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Close'
            },
            'intent': event['sessionState']['intent']
        },
        'messages': [message],
        'sessionId': event['sessionId'],
        'requestAttributes': event['requestAttributes'] if 'requestAttributes' in event else None
    }
    
def get_session_attributes(event):
    sessionState = event['sessionState']
    if 'sessionAttributes' in sessionState:
        return sessionState['sessionAttributes']

    return {}
