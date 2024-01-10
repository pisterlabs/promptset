import requests
import json
import openai
import Keps

openai.api_key = Keps.gpt_api_key

def analyze_news(some_news):
    URL = "https://api.openai.com/v1/chat/completions"

    payload = {
    "model": "gpt-3.5-turbo-1106",
    "messages": [{"role": "user", "content": some_news}],
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

    response_content = response.content.decode('utf-8')  # Decode bytes to string
    response_json = json.loads(response_content)  # Parse string as JSON

    # Access the content field
    content = response_json["choices"][0]["message"]["content"]
    return content

print(analyze_news("Analyze the following news and briefly predict its impact on the stock price: will it increase, decrease, or have no effect? Limit your response to 5-6 words and output to be informat of Stock names: Positive, negative or neutral. Rivian Stock Falls As EV Deliveries Lag As Tesla Beats Expectations"))