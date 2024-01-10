from langchain.prompts import ChatPromptTemplate

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

import os

llm = HuggingFaceEndpoint(
    endpoint_url=os.environ['LLM_ENDPOINT'],
    huggingfacehub_api_token=os.environ['HF_TOKEN'],
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512
    }
)

model = ChatHuggingFace(llm=llm)

# This function shows the process of creating an output parser
# PLEASE DO NOT edit this function
def get_basic_output_parser():
    # Set up an output parser, starting with schema
    is_food_schema = ResponseSchema(name = "is_food", description = "Is the topic a food? Answer True if yes, False if no.")
    # This is a small example, so we just need one schema, 
    # but for more complex outputs, we can include multiple schemas
    response_schemas = [is_food_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    return output_parser

# This function shows the process of creating a prompt 
# that is compatible with the output_parser
def get_basic_prompt():
    # Set up a prompt to extract the necessary information and 
    # include a placeholder for some formatting instructions. 
    # We need the output to be JSON.
    prompt_template = """
    For the following topic, extract the following information:

    is_food

    topic: {topic}

    The format should explicitly adhere to the following format instructions: {format_instructions}
    No extra information should be given. Do not include an explanation. Just give me the JSON.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt

# This function takes our prompt, model, and output parser and
# invokes them via a chain, thus returning the response
# PLEASE DO NOT edit this function but try invoking it with different topics
# and note the output
def invoke_basic_chain(topic):
    prompt = get_basic_prompt()
    output_parser = get_basic_output_parser()
    chain = prompt | model | output_parser

    # Invoke the chain and parse the output using the output parser
    response = chain.invoke({"topic": topic, "format_instructions":  output_parser.get_format_instructions()})
    return response

# TODO Finish this function by creating an output_parser from
# the following schemas:
# - title
# - is_family_friendly
# - genre
# - run_time
# - year_released
def get_complex_output_parser():
    response_schemas = []
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    return output_parser


# TODO: Finish this function by creating a prompt that specifies
# the fields to extract from the response:
def get_complex_prompt():
    prompt_template = """
    For the following movie, extract the following information:

    ...

    movie: {movie}

    Format Instructions: {format_instructions}

    Ensure the output is valid JSON. Only talk about the specified movie. Do not include other movies.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt

# This function chains the prompt, model, and output_parser together
# PLEASE DO NOT edit this function but try invoking it with different movies
def invoke_complex_chain(movie):
    prompt = get_complex_prompt()
    output_parser = get_complex_output_parser()
    chain = prompt | model | output_parser

    # Invoke the chain and parse the output using the output parser
    response = chain.invoke({"movie": movie, "format_instructions":  output_parser.get_format_instructions()})
    return response