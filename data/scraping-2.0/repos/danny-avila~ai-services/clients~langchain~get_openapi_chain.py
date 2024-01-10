from langchain.tools import OpenAPISpec, APIOperation
from langchain.chains import OpenAPIEndpointChain
from langchain.requests import Requests
from langchain.llms import OpenAI

def get_openapi_chain(api_key, model_name):
    spec = OpenAPISpec.from_url(
        "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/")
    operation = APIOperation.from_openapi_spec(
        spec, '/public/openai/v0/products', "get")
    llm = OpenAI(openai_api_key=api_key, model_name=model_name)  # Load a Language Model

    chain = OpenAPIEndpointChain.from_api_operation(
        operation,
        llm,
        requests=Requests(),
        verbose=True,
        return_intermediate_steps=True,  # Return request and response text
        # raw_response=True # Return raw response
    )

    return chain
