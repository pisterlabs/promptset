# this is purely for testing purposes of the langchain serving
from langchain import LLMChain, OpenAI, SerpAPIWrapper
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
import openai
import numpy as np
import os
import uvicorn
from dotenv import load_dotenv
import os
from collections import deque
from typing import Dict, List, Optional, Any
import logging
import json

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
# Langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain ,LLMCheckerChain
from langchain.callbacks import wandb_tracing_enabled
from langchain.prompts import (
	PromptTemplate,
	ChatPromptTemplate,
	HumanMessagePromptTemplate,
)
from langchain.prompts.few_shot import FewShotPromptTemplate

from typing import Optional
from langchain.chains import SimpleSequentialChain ,SequentialChain
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.agents import AgentType, initialize_agent,AgentExecutor
from langchain.tools import tool
from langchain.chains.openai_functions import (
	create_openai_fn_chain,
	create_structured_output_chain,
)
from langchain.schema import HumanMessage, AIMessage, ChatMessage
# from lcserve import serving
import requests 

# # Azure Blob
# from azure.storage.blob import BlobServiceClient

# from datetime import datetime, timedelta
# import json

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response


# App
app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["GET", "POST"],
	allow_headers=["Authorization", "Content-Type"],
	max_age=86400,
)

# ALLOWED_ORIGINS = ['*'],
# ALLOWED_METHODS = '*',
# ALLOWED_HEADERS = '*'

# def check_routes(request: Request):
# 	# Using FastAPI instance
# 	url_list = [
# 		route.path
# 		for route in request.app.routes
# 		if "rest_of_path" not in route.path
# 	]
# 	if request.url.path not in url_list:
# 		return JSONResponse({"detail": "Not Found"})

# Handle CORS preflight requests
# @app.options("/{rest_of_path:path}")
# async def preflight_handler(request: Request, rest_of_path: str) -> Response:
# 	response = check_routes(request)
# 	if response:
# 		return response

# 	response = Response(
# 		content="OK",
# 		media_type="text/plain",
# 		headers={
# 			"Access-Control-Allow-Origin": ALLOWED_ORIGINS,
# 			"Access-Control-Allow-Methods": ALLOWED_METHODS,
# 			"Access-Control-Allow-Headers": ALLOWED_HEADERS,
# 		},
# 	)
# 	return response

# Add CORS headers
# @app.middleware("http")
# async def add_cors_header(request: Request, call_next):
# 	response = check_routes(request)
# 	if response:
# 		return response

# 	response = await call_next(request)
# 	response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGINS
# 	response.headers["Access-Control-Allow-Methods"] = ALLOWED_METHODS
# 	response.headers["Access-Control-Allow-Headers"] = ALLOWED_HEADERS
# 	return response

# app.add_middleware(
# 	CORSMiddleware,
# )

MODEL_NAME = "gpt-3.5-turbo-0613"
TEMPERATURE = 0.0

MODEL_NAME = "gpt-3.5-turbo-0613"
TEMPERATURE = 0.0

class VideoGenerator:
	
	# task_list: deque = Field(default_factory=deque)
	video_chain2: Chain = Field(default_factory=Chain)
	
	def __init__(self,llm):

		expire_prompt = PromptTemplate(
			template="""You are an expiratity date estimator and food ingrediant identifier, given this list of inputs of food ingredients,1.Identify the food ingrediants E.g Bread, Apple, Milk, Remove non-food items E.g bottle,package and 2.extimate when the food ingredients will expire in No.days E.g 30days 7days 90days.Then convert it to number of days.

			Some Examples of ingrediants to estimate their expiration date:\n\nmy inventory of ingrediants:\n{ingredients}\n\nTask:1.Remove all non-food ingrediants E.g Bottle,package  2.Use only these names above to estimate expiration date of each ingrediant.""",
			input_variables= ["ingredients"]
		)
		recipe_prompt = PromptTemplate(
	template="""You are chef writing down steps based on food details. You have to write steps for every dishes based on the each details given. Each dish name is "label".The dishes are: {recipe_details}\nGenerate steps for each dish notated as one json object in the list""",
	input_variables= ["recipe_details"]
)
		json_schema= {
	"name": "estimate expiration date",
	"description": "Given ingrediants and their Quantity, estimate the expiration date of each ingrediant",
	"type": "object",
	"properties": {
	  "list_of_ingrediants": {
		"type": "array",
		"items": {
		  "type": "object",
		  "properties": {
			"Name": { "type": "string","description":"Only extract food ingrediants." },
			"base_name" : { "type": "string","description":"Only the base food ingrediant name"},
			"Quantity": { "type": "integer" },
			"Days_to_expire": { "type": "integer","description":"Estimated days the ingrediant will expire in number of days E.g 30days, 14days"}
		  },
		  "required": ["Name", "Quantity", "Expiration"]
		}
	  }
	},
	"required": ["list_of_ingrediants"]
  }
		recipe_json_schema= {
			"name": "Write recipe Steps",
			"description": "writing down steps based on food details",
			"type": "object",
			"properties": {
			"list_of_ingrediants": {
				"type": "array",
				"items": {
				"type": "object",
				"properties": {
					"index": { "type": "integer" },
					"label": { "type": "string","description":"Name of the dish based on details given. Do not change from details" },
					"steps": { "type": "string","description":"Steps: 1.\n2.\n3.\n4\n5.\n (and so on till the end of the recipe)"}
				},
				"required": ["index", "label", "steps"]
				}
			}
			},
			"required": ["list_of_ingrediants"]
		}
		self.expire_chain = create_structured_output_chain(json_schema, llm, expire_prompt, verbose=True)

		tools = [
			Tool(
				name = "extract_ingredients_estimate_expiry_date",
				func=self.expire_chain.run,
				description="identify/extract only food ingredients and for only those food ingrediants estimate expiry date in number of days. Output JSON",
				
			),
		]

		prefix = """You are an AI who performs one task based on the following objective:1. Remove non-food ingrediants E.g bottle,package,extract only food ingrediants E.g Bread, Apple, Milk 2. Estimate the expiry date of each food ingrediant in number of days E.g 30days 7days 90days.3. Parse it back as JSON format"""
		suffix = """Question: {task} Return JSON format using outputparser
		{agent_scratchpad} if Observation starts with: 'list_of_ingrediants:\nreuturn at json'"""
		prompt = ZeroShotAgent.create_prompt(
			tools,
			prefix=prefix,
			suffix=suffix,
			input_variables=["task", "agent_scratchpad"],
		)
		self.agent_expire = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
		self.recipe_step_chain = create_structured_output_chain(recipe_json_schema, llm, recipe_prompt, verbose=True)
	  
	
	def expire(self,ingredients):
		results = self.expire_chain.run(ingredients=ingredients)
		return results
	
	def RecipeSteps(self,recipe_details):
		results = self.recipe_step_chain.run(recipe_details = recipe_details)
		return results


class InputModel(BaseModel):
	input: str

@app.get("/")
async def root() -> dict:
	return {"message": "Hello World"}

@app.post("/expire")
async def expire(inputBody: InputModel) -> dict:
	input = inputBody
	headers = {'Access-Control-Allow-Headers': 'Content-Type',
			   'Access-Control-Allow-Origin': '*',
			   'Access-Control-Allow-Methods': '*'}
	videoGenerator = VideoGenerator(llm=ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE))
	print("TYPE:", type(input), input)
	output = videoGenerator.expire(input)
	# Example /expire
	"""
	{'list_of_ingrediants': [{'Name': 'ENR WHT BREAD 600G',
	'base_name': 'Bread',
	'Quantity': 2,
	'Days_to_expire': 30},
	{'Name': 'PASAR CHINA YA PEAR',
	'base_name': 'Pear',
	'Quantity': 3,
	'Days_to_expire': 7},
	{'Name': 'FP RED SPINACH',
	'base_name': 'Spinach',
	'Quantity': 1,
	'Days_to_expire': 90},
	{'Name': 'Oranges',
	'base_name': 'Orange',
	'Quantity': 5,
	'Days_to_expire': 14}]}
	"""
	# NER model for ingrediants

	# Honestly this can change
	return JSONResponse(content = output, headers = headers)

@app.post("/RecipeSteps")
async def expire(inputBody: InputModel) -> list:
	input = json.loads(inputBody.input)
	headers = {'Access-Control-Allow-Headers': 'Content-Type',
			   'Access-Control-Allow-Origin': '*',
			   'Access-Control-Allow-Methods': '*'}
	videoGenerator = VideoGenerator(llm=ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE))
	# remove "image_object" key from each recipe in input list of dict
	temp_image_object = []
	for i, recipe in enumerate(input):
		temp_image_object.append(recipe["image_object"])
		del recipe["image_object"]
	
	
	output = videoGenerator.RecipeSteps(input)
	print('output for Recipe Generation: ', output)
	# Add "steps" key to each recipe in input list of dict
	for i, recipe in enumerate(input):
		print("RECIPE:", recipe)
		print(f'added steps to recipe: {recipe["label"]}')
		# Add new "steps" key to output
		recipe["steps"] = output["list_of_ingrediants"][i]["steps"]
		# add back "image_object" key to each recipe in input list of dict
		recipe["image_object"] = temp_image_object[i]
	
	"""
	Example Final output, same as input with extra "steps" key
	"""
	return JSONResponse(content = input, headers = headers)

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# # Set up CORS
# origins = [
# 	"*"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/RecipeSteps")
# def read_root():
#     return {"Hello": "World"}

if __name__ == '__main__':
	# Listening on all network interfaces: 0.0.0.0
	import uvicorn
	uvicorn.run("app:app", host="0.0.0.0", port=8080)