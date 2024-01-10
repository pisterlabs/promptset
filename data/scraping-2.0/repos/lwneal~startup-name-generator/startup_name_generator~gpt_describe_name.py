import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def describe_name(name):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {
          "role": "system",
          "content": "You generate startup ideas. Receive a name and describe it as a business. Be very consise."
        },
        {
          "role": "user",
          "content": "whatevershout.com"
        },
        {
          "role": "assistant",
          "content": "Like Twitter, but with no pretensions"
        },
        {
          "role": "user",
          "content": "trailpink.com"
        },
        {
          "role": "assistant",
          "content": "hiking for breast cancer awareness"
        },
        {
          "role": "user",
          "content": name,
        },
        ],
        temperature=1,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message['content']
