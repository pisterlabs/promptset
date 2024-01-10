# from flask import Blueprint, jsonify
# from langchain import OpenAI, PromptTemplate, LLMChain, SequentialChain
# import os

# generate_response_bp = Blueprint('generate_response', __name__)

# key = os.environ['OPENAI_API_KEY']

# @generate_response_bp.route("/generate_response", methods=["GET"])
# def generate_response():
#     # Create the first LLMChain that says "Who are you?"
#     llm = OpenAI(model_name="text-davinci-003", temperature=.7, openai_api_key=key)
#     template = "Who are you?"
#     prompt_template = PromptTemplate(input_variables=[], template=template)
#     chain1 = LLMChain(llm=llm, prompt=prompt_template, output_key="name")

#     # Create the second LLMChain that takes the output from the first chain and prompts "What is this person like?"
#     llm = OpenAI(temperature=.7, openai_api_key=key)
#     template = "What is this person like? {name}"
#     prompt_template = PromptTemplate(input_variables=["name"], template=template)
#     chain2 = LLMChain(llm=llm, prompt=prompt_template, output_key="description")

#     # Create the SequentialChain that runs the two LLMChains in sequence
#     overall_chain = SequentialChain(chains=[chain1, chain2],
#                                     input_variables=[],
#                                     output_variables=["name", "description"],
#                                     verbose=True)

#     response = overall_chain({})
#     return jsonify(response)
