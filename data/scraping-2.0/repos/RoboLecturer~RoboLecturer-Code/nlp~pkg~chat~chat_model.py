# This script is where the calls to the openai models are

import openai
import time
import sys
sys.path.append('/Users/busterblackledge/')
from keys import openai_API_key
openai.api_key=openai_API_key

def getModel(a):
    """get the current model under use
    @returns: model [string]
    """
    if a == 1:
        return "chatgpt"
    elif a == 2:
        return "davinci"

def chatGPT(query):
    """call the openai chatGPT api
    @params: query [string]
    @reutrns: response [string]
    """
    # create completion
    completions = openai.ChatCompletion.create(
		    model="gpt-3.5-turbo",
		    messages=[
				    {"role": "system", "content": f"{query}"},
			    ]
	    )
    response = completions['choices'][0]['message']['content']
    return response

def daVinci(query):
    """call the openai daVinci api
    @params: query [string]
    @returns: response [string]
    """
    # create completion
    completions = openai.Completion.create(
		engine = "text-davinci-003",
		prompt = f"{query}",
		max_tokens = 1024,	
		n = 1,
		temperature = 0.2,
	)
    response = completions.choices[0]["text"]
    return response

def getResponse(query):
    """get a response to the query using the selected chat model
    @params: query [string]
    @returns: response [string]
    """
    model = getModel(1)
    try:
        if model == "chatgpt":
            response = chatGPT(query)
        elif model == "davinci":
            response = daVinci(query)
    except:
        model = getModel(2)
        if model == "chatgpt":
            response = chatGPT(query)
        elif model == "davinci":
            response = daVinci(query)

    return response

def getEmbedding(content):
    """Get embedding for query from ada for use with PineCone
    @params:
        content: [string] - text and its id
    @returns:
        embeds: list|float - list of embedding vectors
    """
    content = content.encode(
        encoding = 'ASCII',
        errors = 'ignore'
    ).decode()

    model = 'text-embedding-ada-002'
    # text  = [x['text'] for x in content]
    text = content 

    try:
        response = openai.Embedding.create(
            input = text,
            engine = model
        )
    except: 
        done = False
        while not done:
            time.sleep(1)
            try:
                response = openai.Embedding.create(input=text, engine=model)
                done = True
            except:
                pass
    # create list of embeddings
    embeds = [record['embedding'] for record in response['data']]

    return embeds

def flatternConvo(conversation):
    """flattern a list of conversation elements
    @params: conversation: list|dict{}|string - list of the conversation strings
    @returns: convo [string] - single string of the conversation
    """
    convo=""
    for i in conversation:
        convo += '%s: %s\m' % (i['role'].upper(), i['content'])
    return convo.strip()
