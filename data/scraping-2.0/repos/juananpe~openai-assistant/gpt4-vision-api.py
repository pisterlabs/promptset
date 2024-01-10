from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": "https://img.jagranjosh.com/images/2022/November/28112022/Spot-7-Differences-in-15-Seconds.webp",
                },
                {
                    "type": "text",
                    "text": "Can you spot 7 differences between the image in the left and the one in the right?",
                },
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0])