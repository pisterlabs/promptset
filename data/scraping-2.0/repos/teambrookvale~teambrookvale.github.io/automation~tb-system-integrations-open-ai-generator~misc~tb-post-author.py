import os
import openai
import json

# Load your API key from an environment variable or secret management service
openai.api_key = "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"

system_1 = 'Passion.io'
system_2 = 'Google Meet'


prompt = f'''
Write blog post in HTML and cover the following:
 - {system_1}
 - {system_2}   
 - Integration of the two through API or SDK
 - Problems their integration solves
 - Conclusion
'''
messages=[{"role": "user", "content":prompt}]

response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

with open("passion-io-google-meet.json", "w") as file:
    file.write(json.dumps(response))

with open("passion-io-google-meet.html", "w") as file:
    file.write(response.choices[0].message.content)
