from openai import OpenAI

client = OpenAI("key")
response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": """Convert the 2 tables (example 1, example 2) to a csv. Only respond with the csv and don't include newline characters in your response"""},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://i.ibb.co/f0CgXmv/Screenshot-2023-11-22-142205.png",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])