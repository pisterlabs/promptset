# This script is where the calls to the openai models are
import openai
import sys
sys.path.append('/Users/busterblackledge')
from keys import openai_API_key

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
    openai.api_key = openai_API_key
    # create completion
    completions = openai.ChatCompletion.create(
		    model="gpt-3.5-turbo",
		    messages=[
                    {'role': 'user',
                     'content': f"{query}"}
            ]
	    )
    response = completions['choices'][0]['message']['content']
    return response

def daVinci(query):
    """call the openai daVinci api
    @params: query [string]
    @returns: response [string]
    """
    openai.api_key = openai_API_key
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
            print("chatGPT failed - Using GPT-3")
    except:
        model = getModel(2)
        if model == "chatgpt":
            response = chatGPT(query)
        elif model == "davinci":
            response = daVinci(query)
            print("chatGPT failed - Using GPT-3")

    return response

