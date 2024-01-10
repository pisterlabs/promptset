import functools
import requests
import os


def huggingface(model_key, prompt: str, temperature=0.7, max_tokens=120, stop=None):
	API_URL = "https://api-inference.huggingface.co/models/" + model_key # "bigscience/bloom"
	response = requests.post(API_URL, headers={
		"Authorization": "Bearer " + os.environ['HUGGINGFACE_API_KEY'],
	}, json={
		"inputs": prompt,
		"parameters": {
			"max_length": max_tokens,
			"temperature": temperature,
			"return_full_text": False,
			"stop": stop,
		}
	})
	j = response.json()
	if type(j) is list:
		j = j[0]
	return j['generated_text']

def openai(model_key, prompt: str, temperature=0.7, max_tokens=120, stop=None) -> str:
	body = {
		'model': model_key,
		'prompt': prompt,
		'temperature': temperature,
		'max_tokens': max_tokens,
		'top_p': 1,
		'frequency_penalty': 0,
		'presence_penalty': 0,
		'stop': stop,
	}
	headers = {
		'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'],
		'Content-Type': 'application/json',
	}
	response = requests.post('https://api.openai.com/v1/completions', json=body, headers=headers)
	response = response.json()

	print("POST https://api.openai.com/v1/completions")
	print(f"{body=} {headers=}")
	print(f"{response=}")

	if 'choices' not in response:
		raise ValueError("Invalid response from OpenAI: " + str(response))

	text = response['choices'][0]['text']
	prompt_tokens = response['usage']['prompt_tokens']
	completion_tokens = response['usage']['completion_tokens']

	return {
		"text": text,
		"prompt_tokens": prompt_tokens,
		"completion_tokens": completion_tokens,
	}

def openai_embeddings(text: str) -> list[float]:
	body = {
		'model': 'text-embedding-ada-002',
		'input': text,
	}
	headers = {
		'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'],
		'Content-Type': 'application/json',
	}
	response = requests.post('https://api.openai.com/v1/embeddings', json=body, headers=headers)
	response = response.json()

	print("POST https://api.openai.com/v1/embeddings")
	print(f"{body=} {headers=}")
	print(f"{response=}")

	if 'data' not in response:
		raise ValueError("Invalid response from OpenAI: " + str(response))

	return response['data'][0]['embedding']

def get_api(method: str):
	if method.startswith("hf:"):
		return functools.partial(huggingface, method[3:])
	elif method.startswith("openai:"):
		return functools.partial(openai, method[7:])
	else:
		return None
