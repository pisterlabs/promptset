# get_emotion.py
import openai
import re
import requests
import json

API_KEY = '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'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

def get_emotion(text):
    prompt = f"Analyze the emotions of the following tweet based on Plutchik's wheel of emotions. Strictly only return 3 comma separated values that are primary, secondary, and tertiary emotions in that order: \"{text}\""

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You will strictly return 3 comma separated values."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50,
        "n": 1,
        "stop": None,
        "temperature": 0.5,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, data=json.dumps(data))

    if response.status_code == 200:
        choices = response.json()["choices"]
        emotions = choices[0]["message"]["content"].strip().split(', ')

        return {
            'primary_emotion': emotions[0] if len(emotions) > 0 else None,
            'secondary_emotion': emotions[1] if len(emotions) > 1 else None,
            'tertiary_emotion': emotions[2] if len(emotions) > 2 else None,
        }
    else:
        print("Error:", response.status_code, response.text)
        return None
  
