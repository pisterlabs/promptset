import openai
import requests
import json
msg='''suggest five various dishes and their recipes based on the ingredients ->
    
    name:'';
    ingredients:'';
    recipe:'';
    
 '''
def sum(text):
    openai.api_key = 'sk-******************************'
    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": msg+str(text)}],
    "temperature" : 1.0,
    "top_p":1.0,
    "n" : 1,
    "stream": False,
    "presence_penalty":0,
    "frequency_penalty":0,
    }

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai.api_key}"
    }

    response = requests.post(URL, headers=headers, json=payload, stream=False)

    data=json.loads(response.content)

    summary = data['choices'][0]['message']['content']
    print(summary)
    return summary
    