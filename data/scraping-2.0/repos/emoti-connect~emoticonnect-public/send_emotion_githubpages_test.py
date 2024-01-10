import openai
import json
import os   
import requests


openai.api_key = os.environ['openai_key']

def chatgpt_inference(messages, model="gpt-3.5-turbo", max_tokens=256, temperature=0.2):
  response = openai.ChatCompletion.create(
      model=model,
      max_tokens=max_tokens,
      temperature=temperature,
      messages=messages,
  )
  return  response



def lambda_handler(event, context):
    if("key" in event and event["key"]=="0987654321"):
        model = "gpt-3.5-turbo"
        messages = json.loads('[{"role": "user","content":"Say nothing."}]')
        max_tokens = 256      
        temperature=0.3

        if("model" in event):
            model = event["model"]

        if("apple_watch_text" in event):
            messages = json.loads("""[{
        "role": "system",
        "content": "We are taking speech from one person and converting it into emotion for neurodivergent individual to understand the emotions taking place in the conversation (use this context to derive emotions from the text for the neurodivergent person)"
        },{
        "role": "user",
        "content": 
        "Emotion: Happiness\\nHappiness Intensity: [ChatGPTs assesment of happiness intensity on a scale of 1-10]\\nHappiness Confidence: [ChatGPTs confidence level for detecting happiness as a percentage]\\n\\nEmotion: Sadness\\nSadness Intensity: [ChatGPTs assesment of sadness intensity on a scale of 1-10]\\nSadness Confidence: [ChatGPTs confidence level for detecting sadness as a percentage]\\n\\nEmotion: Anger\\nAnger Intensity: [ChatGPTs assesment of angerintensity on a scale of 1-10]\\nAnger Confidence: [ChatGPT confidence level for detecting anger as a percentage]\\n\\nEmotion: Fear\\nFear Intensity: [ChatGPTs assesment of fear intensity on a scale of 1-10]\\nFear Confidence: [ChatGPSs confidence level for detecting fear as a percentage]\\n\\nEmotion: Neutral\\nNeutral Intensity: [ChatGPTs assesment of neutralintensity on a scale of 1-10]\\nNeutral Confidence: [ChatGPTs confidence level for detecting neutrality as a percentage]\\n\\nPrimary Emotion: [ChatGPTs calculation of the most proeminent emotion based on confidence and intensity of the five emotions]\\nSecondary Emotion: [ChatGPTs calculation of the second most proeminent emotion based on confidence and intensity of the five emotions]\\nBalanced or Not Balanced: [ChatGPTs assesment whether the primary and secondary emotions are balanced or not balanced (1 means balanced, 0 means not balanced)]"
        },{
        "role": "user",
        "content": "Can you provide an answer in a json format for the following text:"
        }]""")
            messages[2]["content"] += event["apple_watch_text"]
        
    
        if("messages" in event):
            messages = event["messages"]

        if("max_tokens" in event):
            max_tokens = event["max_tokens"]
        
        if("temperature" in event):
            temperature = event["temperature"]
        
        response = chatgpt_inference(messages = messages, model = model, max_tokens=max_tokens, temperature=temperature)
        
        # Process the OpenAI API response to extract emotions, intensity, and confidence
        processed_data = process_response(response)

        # Send the processed data to the GitHub webpage
        send_to_github(processed_data)   
        
        return {
        'statusCode': 200,
        'body': response
        }
    else:
        return {
        'statusCode': 401,
        'body': "Unauthorized"
        }
      
      
def process_response(response):
    # Process the OpenAI API response and extract emotions, intensity, and confidence
    # Modify this function based on the structure of the OpenAI API response
    emotions = response['emotions']
    intensity = response['intensity']
    confidence = response['confidence']

    processed_data = {
        'emotions': emotions,
        'intensity': intensity,
        'confidence': confidence
    }

    return processed_data

def send_to_github(data):
    github_api_url = 'https://api.github.com/repos/EmotiConnect/EmotiConnect.github.io/contents/main/data.json'
    github_username = 'EmotiConnect'
    github_repository = 'EmotiConnect.github.io'
    github_path = 'main/data.json'

    # Create the API URL with the appropriate username, repository, and file path
    url = github_api_url.format(username=github_username, repository=github_repository, path=github_path)

    # Prepare the data payload to be sent to GitHub
    payload = {
        'message': 'Update data file',
        'content': base64.b64encode(json.dumps(data).encode('utf-8')).decode('utf-8'),
        'branch': 'main'  # Replace with the branch name where your file is located
    }

    # Set the authentication headers (if required)
    headers = {
        'Authorization': 'Bearer github_token',  # Replace with your GitHub personal access token
        'Content-Type': 'application/json'
    }

    # Send the data payload to the GitHub API
    response = requests.put(url, headers=headers, json=payload)

    # You can handle the response from the GitHub API here
    # For example, check the status code or response content

    return response      
