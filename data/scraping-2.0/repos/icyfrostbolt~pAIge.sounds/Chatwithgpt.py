import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY3")  # API key
#openai.api_key =

def chat_with_gpt(summary):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "make a playlist of 15 real songs with just song titles and " +
                                              "artists that match the vibe of the following summary:" +
                                              summary +
                                              "in json format named playlist, organized with \'title\' and artist" +
                                              ". Only include JSON in your response " +
                                              "with no other words"}],
        max_tokens=1024)
    content = (response.choices[0].message.content)
    #print(content)
    return content

