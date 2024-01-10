#!/usr/bin/python3
# This script connects to OpenAI and continues the story.

import os
import openai
import connix
import time
import run
import boto3
import json

print(connix.header())
form = connix.form()

if not run.validate(False):
	run.error("Unauthorized.")

if 'prompt' not in form:
	run.error("Missing field.")

prompt = form['prompt']
username = ""
if "HTTP_COOKIE" in os.environ and "dcpim_net_session=" in os.environ['HTTP_COOKIE']:
	username = connix.in_tag(os.environ['HTTP_COOKIE'], "dcpim_net_session=", ";")
if username == "":
	username = "Guest"
world = ""
if "world" in form:
	world = "Keep in mind that {}".format(form['world'])

def chatgpt(context, prompt):
	global world
	context = "Continue the following story. " + world + " Keep your answer to about one sentence.\n\n" + context + " "
	output = ""
	attempts = 0
	try:
		while output == "":
			completion = openai.Completion.create(engine=run.config('OPENAI_MODEL'), prompt=context + prompt, temperature=0.8)
			response = completion.choices[0].text
			output = response.replace('\n','').replace('.','. ').lstrip()
			if output == "":
				time.sleep(1)
				attempts = attempts + 1
			if attempts > 5:
				return "<b><i>Failed to load.</i></b> {}".format(output)
	except Exception as err:
		return "<b><i>Failed to load.</i></b> {}".format(err)
	return output

def titan(context, prompt):
	global world
	context = "Continue the following story. " + world + " Keep your answer to about one sentence.\n\n" + context + " "
	output = ""
	try:
		bedrock = boto3.client("bedrock-runtime", region_name=run.config("AWS_REGION"))
		body = json.dumps({
			"inputText": context + prompt,
			"textGenerationConfig": {
				"maxTokenCount": 512,
				"stopSequences": [],
				"temperature": 0,
				"topP": 0.9
			}
		})
		response = bedrock.invoke_model(
			body = body,
			modelId = "amazon.titan-text-express-v1",
			accept = "application/json",
			contentType = "application/json"
		)
		raw = json.loads(response.get('body').read())
		output = raw.get('results')[0].get('outputText').replace('\n','').replace('.','. ').lstrip()
	except Exception as err:
		return "<b><i>Failed to load.</i></b> {}".format(err)
	return output

#
# ChatGPT
#

# Get previous sentence
data = run.query("SELECT sentence FROM ai_story WHERE username != 'Titan' ORDER BY id DESC LIMIT 1;")
if data[0][0][0] == '<':
	previous = ""
else:
	previous = data[0][0]

# Add prompt to db
run.sql("INSERT INTO ai_story (username, sentence, world, date) VALUES (%s, %s, %s, %s);", username, prompt, world.replace('Keep in mind that ',''), connix.now())

# Fetch model output
openai.api_key = run.config('OPENAI_KEY')
output = chatgpt(previous, prompt)

# Add output to db
if output[0] == "<":
	run.sql("INSERT INTO ai_story (username, sentence, world, date) VALUES (%s, %s, %s, %s);", "System", output, world.replace('Keep in mind that ',''), connix.now())
else:
	run.sql("INSERT INTO ai_story (username, sentence, world, date) VALUES (%s, %s, %s, %s);", "ChatGPT", output, world.replace('Keep in mind that ',''), connix.now())

#
# Titan
#

# Get previous sentence
data = run.query("SELECT sentence FROM ai_story WHERE username != 'ChatGPT' ORDER BY id DESC LIMIT 1;")
if data[0][0][0] == '<':
	previous = ""
else:
	previous = data[0][0]

# Fetch model output
output = titan(previous, prompt)
if output[0] == "<":
	run.sql("INSERT INTO ai_story (username, sentence, world, date) VALUES (%s, %s, %s, %s);", "System", output, world.replace('Keep in mind that ',''), connix.now())
else:
	run.sql("INSERT INTO ai_story (username, sentence, world, date) VALUES (%s, %s, %s, %s);", "Titan", output, world.replace('Keep in mind that ',''), connix.now())

# Add output to db

# Done
run.done()
