import json
import uuid
import os
import traceback
import boto3

import langchain

from langchain.llms.bedrock import Bedrock
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.cache import InMemoryCache

langchain.llm_cache = InMemoryCache()
langchain.debug = True
langchain.verbose = True

POLLY = boto3.client("polly")
TRANSLATE = boto3.client("translate")
KENDRA_CLIENT = boto3.client("kendra")
BEDROCK_CLIENT = boto3.client("bedrock")
KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID", "dc60affd-f6aa-4ae2-9b4b-a0dc7da5bbb6")

MAX_TOKENS_TO_SAMPLE: int = 2048
TEMPERATURE: float = 0.9
TOP_K: int = 250
TOP_P: float = 0.99
STOP_SEQUENCES = ["\n\nHuman:"]

MAX_HISTORY_LENGTH = 10
DEFAULT_PROMPT_TEMPLATE = """\
Human: This conversation demonstrates the power of retrieval augmented generation.
You are the AI. You will synthesize a response from your own context and the provided documents.
Follow these rules for your responses:
* If the documents are relevant, use them to synthesize your answer.
* DO NOT MENTION THE DOCUMENTS IN YOUR ANSWER.
* For simple factual questions, respond with a single sentence.
* For more complex questions that reference specific information, you can respond with 1000s of tokens. Use complex formatting like markdown.
* If you don't know the answer to a question, truthfully reply: "I'm not sure how to answer that, but I'd be happy to introduce you to a Caylent engineer who can help."
* Reply in the language the question was asked in.

Please confirm your understanding of the above requirements before I pass in the user's question and documents.

Assistant: I understand the requirements. I will synthesize responses based on the provided documents without mentioning that I am doing this. I will use simple sentences for factual questions and complex markdown formatting for more involved questions. If I don't know the answer, I will direct you to a Caylent engineer. I will match the language of the question. Please provide the question and documents when ready.

Human: Here are potentially relevant documents in <documents> XML tags:
<documents>
{context}
</documents>
Answer the user's question: "{question}"

Assistant:"""

DEFAULT_CONDENSE_QA_TEMPLATE = """\
Human: Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

"""
Request JSON format for proxy integration
{
	"resource": "Resource path",
	"path": "Path parameter",
	"httpMethod": "Incoming request's method name"
	"headers": {Incoming request headers}
	"queryStringParameters": {query string parameters }
	"pathParameters":  {path parameters}
	"stageVariables": {Applicable stage variables}
	"requestContext": {Request context, including authorizer-returned key-value pairs}
	"body": "A JSON string of the request payload."
	"isBase64Encoded": "A boolean flag to indicate if the applicable request payload is Base64-encode"
}


Response JSON format
{
	"isBase64Encoded": true|false,
	"statusCode": httpStatusCode,
	"headers": { "headerName": "headerValue", ... },
	"body": "..."
}

"""


def response_proxy(data):
    """
    For HTTP status codes, you can take a look at https://httpstatuses.com/
    """
    response = {}
    response["isBase64Encoded"] = False
    response["statusCode"] = data["statusCode"]
    response["headers"] = {}
    if "headers" in data:
        response["headers"] = data["headers"]
    response["body"] = json.dumps(data["body"])
    return response


def request_proxy(data):
    request = {}
    request = data
    if data["body"]:
        request["body"] = json.loads(data["body"])
    return request


def handle_query(event):
    """
    This is the main function for handling the query
    """
    return "Placeholder"


MAX_HISTORY_LENGTH = 10


def build_chain(prompt_template, condense_qa_template):
    llm = Bedrock(
        client=BEDROCK_CLIENT,
        model_id="anthropic.claude-v2",
        model_kwargs={
            "max_tokens_to_sample": MAX_TOKENS_TO_SAMPLE,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "stop_sequences": STOP_SEQUENCES,
        },
    )

    retriever = AmazonKendraRetriever(
        client=KENDRA_CLIENT, index_id=KENDRA_INDEX_ID, top_k=3
    )

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=standalone_question_prompt,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return qa


def run_chain(chain, prompt: str, history=[]):
    return chain({"question": prompt, "chat_history": history})


def run_chatbot_request(request):
    chat_input = request["body"]["chat_input"]

    # May not need this - can just handle it in the UI?
    # question_with_id = {"question": chat_input, "id": len(st.session_state.questions)}

    chat_history = []
    if len(chat_history) == MAX_HISTORY_LENGTH:
        chat_history = chat_history[:-1]

    llm_chain = build_chain(DEFAULT_PROMPT_TEMPLATE, DEFAULT_CONDENSE_QA_TEMPLATE)

    result = run_chain(llm_chain, chat_input, chat_history)
    answer = result["answer"]
    chat_history.append((chat_input, answer))

    document_list = []
    if "source_documents" in result:
        for d in result["source_documents"]:
            if not (d.metadata["source"] in document_list):
                document_list.append((d.metadata["source"]))

    # Return answer which is result, and then document list as sources
    return {"answer": answer, "sources": document_list}


def get_an_answer(request):
    try:
        value = run_chatbot_request(request)
        return value
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def lambda_handler(event, _):
    response = {}
    try:
        response["statusCode"] = 200
        response["headers"] = {}
        """
        Add your key/values to be returned here
        """

        data = {
            "message": "Hello World!",
        }

        response["body"] = data
    except Exception:
        response["statusCode"] = 500
        response["body"] = {}
    return response
