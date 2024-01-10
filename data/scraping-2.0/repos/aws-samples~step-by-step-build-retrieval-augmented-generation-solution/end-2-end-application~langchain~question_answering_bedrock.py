from query_embedding import query
from langchain.chains import LLMChain
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms import Bedrock
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import json
from langchain import PromptTemplate

def build_chain():
    region_name = "<AWS REGION NAME>"
    profile_name = "<AWS CREDENTIAL PROFILE>"

    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            input_str = json.dumps({"inputs" : [[
            {"role" : "user", "content" : prompt}]],
            "parameters" : {**model_kwargs}})
            return input_str.encode('utf-8')

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json[0]["generation"]["content"]

    inference_modifier = {'max_tokens_to_sample':2048, 
        "temperature":0.01,
        "top_k":2,
        "top_p":0.5,
        "stop_sequences": ["\n\nHuman"]
    }

    content_handler = ContentHandler()

    bedrock_llm = Bedrock(
        credentials_profile_name= profile_name,
        model_id="anthropic.claude-v2",
        region_name=region_name,
        model_kwargs = inference_modifier
    )


    prompt_template = """
Human: Use the following retrieved information to provide an accurate and step-by-step answer to the question at the end. If you don't know the answer, just write #IDONTKNOW, don't try to make up an answer.

Retrieved information: {retrieved_information}


Question: {question}

Please provide step-by-step reasoning
Assistant:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["retrieved_information", "question"]
    )

    llmchain = LLMChain(llm=bedrock_llm, prompt=prompt)
    return llmchain

import re

def run_chain(chain, question: str, history=[]):
    search_query = question
    if '##' in question:
        search_query = re.findall(r'##(.*?)##', question)[0]
        question = re.sub(r'##.*?##', '', question)
    result = query(search_query)
    sources = list(map(lambda x: "page " + x[0], result))
    #source_documents = list(map(lambda x: x[1], result))
    source_documents = result
    answer = chain.run({
        'question': question,
        'retrieved_information': source_documents,
        'history':[]
    })
    
    result = {
        'answer': answer,
        'source_documents': source_documents,
        'sources': sources
    }
    return result

if __name__ == "__main__":
    llmchain = build_chain()
    Question = "what is SageMaker?"
    result = run_chain(llmchain, Question)
    print(result)
