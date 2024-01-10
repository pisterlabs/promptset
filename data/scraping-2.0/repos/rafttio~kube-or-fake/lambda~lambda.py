import json
import openai
import os
import random


openai.api_key = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')


def retry_generate_response(messages):
	try:
		response = openai.ChatCompletion.create(
			model=OPENAI_MODEL_NAME,
			messages=messages
		)
		content = response['choices'][0]['message']['content']

		## ensure that ChatGPT's response is in valid JSON
		json.loads(content)
		return content
	except:
		return None

def generate_response(messages):
	while True:
		response = retry_generate_response(messages)
		if response is not None:
			return response

def generate_kube():
    messages = [
        {
            "role": "system",
            "content": """You are a Kubernetes expert.
Generate a random, real Kubernetes term, which is either a resource kind or a field name.
Response format:
{
    \"word\": \"your generated term\",
    \"isReal\": true,
    \"description\": \"a short description of the term, in the form of: <word> is/are ...\"
}"""
        }
    ]
    return generate_response(messages)

def generate_fake():
    messages = [
        {
            "role": "system",
            "content": """You are a Kubernetes expert.
Generate a random, fake Kubernetes term, which resembles a resource kind or a field name.
An average software developer should not be able to immediately realize this is a fake term.
Make sure that this is in fact, not a real Kubernetes term.
Response format:
{
    \"word\": \"your generated term\",
    \"isReal\": false,
    \"description\": \"your description of the term, as if it were real. never mention it is fake or fictional or doesn't exist\"
}"""
        }
    ]
    return generate_response(messages)


def generate_obvious_fake():
    messages = [
        {
            "role": "system",
            "content": """You are a Kubernetes expert.
Generate a random, fake Kubernetes term, which resembles a resource kind or a field name.
An average software developer should be able to immediately realize this is a fake term.
Make sure that this is in fact, not a real Kubernetes term.
Extra points for coming up with a funny term.
Response format:
{
    \"word\": \"your generated term\",
    \"isReal\": false,
    \"description\": \"your description of the term, as if it were real. never mention it is fake or fictional or doesn't exist\"
}"""
        }
    ]
    return generate_response(messages)


def lambda_handler(event, context):
	generators = [generate_kube, generate_fake, generate_obvious_fake]
	random.shuffle(generators)
	body = generators[0]()
	return {
		'statusCode': 200,
		'headers': {
		"Content-Type": "application/json",
			"X-Requested-With": '*',
			"Access-Control-Allow-Headers": 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Requested-With',
			"Access-Control-Allow-Origin": '*',
			"Access-Control-Allow-Methods": 'GET, OPTIONS',
			"Access-Control-Allow-Credentials": True # Required for cookies, authorization headers with HTTPS
		},
		'body': body
	}
