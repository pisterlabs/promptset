from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

client = OpenAI()
response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What are in these images? Is there any difference between them?",
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://contents.mediadecathlon.com/p1875564/k$4eec0a36fb3a9b4124fd59e676fc3a0d/sq/8529877.jpg",
          },
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://contents.mediadecathlon.com/p2254190/k$b6432c2fb00743221776def4824206c1/sq/8558559.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)
print(response.choices[0])