import json
import os

from langchain.prompts import PromptTemplate


ANTHROPIC_INSTANT = 'anthropic.claude-instant-v1'
TITAN_LARGE = 'amazon.titan-tg1-large'
TITAN_EMBED = "amazon.titan-embed-text-v1"

MODEL_IDS = [ANTHROPIC_INSTANT, TITAN_LARGE]

PROMPT_TEMPLATES = {
    (ANTHROPIC_INSTANT, None): """Human: {txt}
    
                                  Assistant:
                               """,

    (ANTHROPIC_INSTANT, 'q&a'): """Human: {txt}
    
                                   Question: {question}
                             
                                   Assistant:
                                """,
    
    (TITAN_LARGE, None): """Command: {txt}"""
}



PROMPT_TEXTS = {
    "blog": "Write me a blog about making strong business decisions as a leader.",
    
    "email": """Write an email from Bob, Customer Service Manager, to the customer "John Doe" 
                who provided negative feedback on the service provided by our customer support engineer""",
    
    "summary": """Please provide a summary of the following text.
                <text>
                AWS took all of that feedback from customers, and today we are excited to announce Amazon Bedrock, \
                a new service that makes FMs from AI21 Labs, Anthropic, Stability AI, and Amazon accessible via an API. \
                Bedrock is the easiest way for customers to build and scale generative AI-based applications using FMs, \
                democratizing access for all builders. Bedrock will offer the ability to access a range of powerful FMs \
                for text and images—including Amazons Titan FMs, which consist of two new LLMs we’re also announcing \
                today—through a scalable, reliable, and secure AWS managed service. With Bedrock’s serverless experience, \
                customers can easily find the right model for what they’re trying to get done, get started quickly, privately \
                customize FMs with their own data, and easily integrate and deploy them into their applications using the AWS \
                tools and capabilities they are familiar with, without having to manage any infrastructure (including integrations \
                with Amazon SageMaker ML features like Experiments to test different models and Pipelines to manage their FMs at scale).
                </text>""",
    
    "q&a": """Use the following pieces of context to provide a concise answer to the question at the end. 
              If you don't know the answer, just say that you don't know, don't try to make up an answer.
              <context>
              {context}
              </context>"""
    
}


BODY_COMPOSERS = {
    ANTHROPIC_INSTANT: lambda prompt: json.dumps({
        "prompt": prompt, 
        "max_tokens_to_sample": 500,
    }),
    
    TITAN_LARGE: lambda prompt: json.dumps({
        "inputText": prompt, 
        "textGenerationConfig":{
        "maxTokenCount":4096,
        "stopSequences":[],
        "temperature":0,
        "topP":0.9
        }})
}


# ==== (Anthropic RAG usage related) ====
_ANTHROPIC_QA_PROMPT_TEMPLATE = """

Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Assistant:"""
ANTHROPIC_QA_PROMPT_TEMPLATE = PromptTemplate(template=_ANTHROPIC_QA_PROMPT_TEMPLATE, 
                                              input_variables=["context", "question"])


# ==== non-streaming response parsers ====
RESPONSE_PARSERS = {
    ANTHROPIC_INSTANT: lambda response_body: response_body.get("completion"),

    TITAN_LARGE: lambda response_body : response_body.get('results')[0].get('outputText')
}


# ===================================================

DEFAULT_PINECONE_API_KEY = os.environ["DEFAULT_PINECONE_API_KEY"]
DEFAUL_PINECONE_ENVIRONMENT = os.environ["DEFAULT_PINECONE_ENVIRONMENT"]
DEFAULT_PINECONE_INDEX_NAME = os.environ["DEFAULT_PINECONE_INDEX_NAME"]

DEFAULT_MODEL_ID = os.environ["DEFAULT_MODEL_ID"]
DEFAULT_EMBEDDINGS_MODEL_ID = os.environ["DEFAULT_EMBEDDINGS_MODEL_ID"]

DEFAULT_INFERENCE_MODIFIER = {
            "max_tokens_to_sample": 4096,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman"]
}
