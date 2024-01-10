# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from google.auth import credentials
from google.oauth2 import service_account
from google.cloud import aiplatform
# from vertexai.preview.language_models import ChatModel, InputOutputTextPair
# from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
import vertexai
from vertexai.preview.language_models import ChatModel, TextGenerationModel, InputOutputTextPair
import json  # add this line

from langchain.chat_models import ChatVertexAI
from langchain.llms import VertexAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain import PromptTemplate, LLMChain

from langchain.vectorstores import MatchingEngine

# Load the service account json file
# Update the values in the json file with your own
with open(
    "credentials.json", encoding="utf-8"
) as f:  # replace 'serviceAccount.json' with the path to your file if necessary
    service_account_info = json.load(f)
    project_id = service_account_info["project_id"]


my_credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = VertexAI(credentials=my_credentials, project=project_id, location="us-central1", model_name="text-bison@001", temperature=0.5, max_output_tokens=1024, top_p=0.95, top_k=40)
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

print(llm_chain.run(question))


# texts = [
#     "The cat sat on",
#     "the mat.",
#     "I like to",
#     "eat pizza for",
#     "dinner.",
#     "The sun sets",
#     "in the west.",
# ]


# vector_store = MatchingEngine.from_components(
#     texts=texts,
#     project_id="project_id",
#     region="us-central1",
#     gcs_bucket_uri="<my_gcs_bucket>",
#     index_id="<my_matching_engine_index_id>",
#     endpoint_id="<my_matching_engine_endpoint_id>",
# )

# vector_store.add_texts(texts=texts)

# vector_store.similarity_search("lunch", k=2)


# chat = ChatVertexAI(credentials=my_credentials, project=project_id, location="us-central1", model_name="codechat-bison@001", temperature=0.5, max_output_tokens=2048, top_p=0.95, top_k=40)

# template = (
#     "You are a helpful assistant who works for Akshay."
# )
# system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# human_template = "{text}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# chat_prompt = ChatPromptTemplate.from_messages(
#     [system_message_prompt, human_message_prompt]
# )

# # get a chat completion from the formatted messages
# a = chat(
#     chat_prompt.format_prompt(text="Write a function that replaces all instances of a string with another string.",
#     ).to_messages()
# )

# print(a)


# # Initialize Google AI Platform with project details and credentials
# aiplatform.init(
#     credentials=my_credentials,
# )

# # Initialize Vertex AI with project and location
# vertexai.init(project=project_id, location="us-central1")



# chat_model = ChatModel.from_pretrained("chat-bison@001")

# parameters = {
#         "temperature": 0.5,
#         "max_output_tokens": 256,
#         "top_p": 0.95,
#         "top_k": 40,
#     }

# chat = chat_model.start_chat(
#         context="You are an experience tour guide in Berlin that can answer any questions a tourist might ask.",
#     )

# response = chat.send_message("my name is sascha I would like to visit berlin can you give me your top 2 recommendations?", **parameters)
# print(f"PaLM 2 response: {response.text}")

