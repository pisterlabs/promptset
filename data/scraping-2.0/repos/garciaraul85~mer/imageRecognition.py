import openai
client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role":"user",
        "content":[
            {"type":"text", "text":"What's in this image?"},
            {
                "type":"image_url",
                "image_url":"https://i.insider.com/562a71f8dd0895a8388b4581?width=1000&format=jpeg&auto=webp"}   
        ]
    }],
    max_tokens=300 
)

print(response.choices[0].message.content)