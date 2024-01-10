import json
import openai
import pdb


def get_api_key():
	# f = open('constants.json')
	# constants = json.load(f)
	# ADD KEY BELOW
	# TODO: fix website data read rights
	openai.api_key = "" #constants["openai_api_key"]
	# f.close()

def make_prompt(elevator_pitch, extras):
  prompt = "here is my elevator pitch:\n" + elevator_pitch
  extras_titles = ["product or service description",
                   "revenue model",
                   "target market",
                   "marketing channels",
                   "competitive analysis",
                   "pricing strategy",
                   "sales strategy",
                   "operational plan",
                   "financial projections",
                   "funding needs",
                   "risk analysis",
                   "exit strategy"]
  for title, input in zip(extras_titles, extras):
    if input == "":
      continue
    prompt += '\n'
    prompt += "my " + title + " is: " + input
  prompt += '\n'
  prompt += 'Can you help me with a good business plan?'
  return prompt

def call_gpt_4(prompt, model="gpt-4", verbose=False):
	if verbose:
			print("__________________________________________________________________")
			print("Input: ", prompt)  
			print("__________________________________________________________________")
			print("Output:")
	get_api_key()
	messages=[{"role": "user", "content": prompt}]    
	completion = openai.ChatCompletion.create(
	  model=model,
	  messages=[
	    {"role": "user", "content": prompt}
	  ]
	)
	response = completion.choices[0].message["content"]

	if verbose:
		print(response)
		print("__________________________________________________________________")
	return response

def web_callable(elevator_pitch, extras, verbose=False):
	prompt = make_prompt(elevator_pitch, extras)
	response = call_gpt_4(prompt, verbose=verbose)
	return prompt, response


# prompt = "I want to create a business selling smart pools. Help me with a good business plan."

# output = call_gpt_4(prompt, verbose=True)