#!/home/jack/Desktop/FlaskAppArchitect_Flask_App_Creator/env/bin/python
import openai
#enter openai key
openai.api_key = "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"
completion = openai.ChatCompletion.create(model="gpt-3", messages=[{"role": "user", "content": "Give me 3 ideas for apps I could build with openai apis "}])
print(completion.choices[0].message.content)