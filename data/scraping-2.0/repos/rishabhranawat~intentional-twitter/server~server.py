# Standard library imports
import io
import os
import threading
import uuid
import warnings

# Third-party library imports
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import pinecone

# constants
OPEN_AI_CHAT_MODEL = 'gpt-3.5-turbo'

# open ai set up
openai.api_key = os.environ['OPENAI_API_KEY']

# pinecone set up
pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
			  environment=os.environ["PINECONE_ENVIRONMENT"])
index = pinecone.Index("arxiv-index-v2")

# app settings
app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={
	r"/*":{
		"origins":"*"
	}
})

def get_gpt_response(content):
	chat_completion = openai.ChatCompletion.create(
		model=OPEN_AI_CHAT_MODEL, 
		messages=[{"role": "user", "content": content}])
	return chat_completion.choices[0].message.content

def get_chatgpt_prompt(query, top_k):
	prompt = f"You are restricted to the data provided in this context. The original query is this: {query}."
	prompt += "Based on the following research author and summary recommend who the author should talk to: \n"
	for each in top_k:
		if 'summary' not in each.metadata: continue
		prompt += f"Title: {each.metadata['title']}; Authors: {each.metadata['authors']}; Summary: {each.metadata['summary']}; url:{each.metadata['url']} \n \n"

	prompt += "Be concise and generate an insightful but factual response."
	prompt += " The structure of the response should be: Title, Author, Url, why they should connect in bullet points. Break it up per relevant research paper."
	return prompt


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

@app.route('/', methods=['GET'])
def root_endpoint():
	return jsonify(
		{
			"message": "Endpoints Documentation",
			"endpoints": {
				"chat": "/api/arxiv-scholar/chat"
			}
		}), 200

@app.route('/api/arxiv-scholar/chat', methods=['POST'])
def chat():
	data = request.get_json()
	query = data['query']
	embedding = get_embedding(query)
	nn = index.query(
		  vector=embedding,
		  top_k=3,
		  include_values=True,
		  include_metadata=True
	)
	prompt = get_chatgpt_prompt(query, nn.matches)
	print(f'Generated Prompt: {prompt}')	
	return jsonify({"response": get_gpt_response(prompt)})

if __name__ == "__main__":
	app.run(debug=True)