#!/usr/bin/python3

print("Content-type: text/html")
print("Access-Control-Allow-Origin: *")
print()

import subprocess
import cgi


form = cgi.FieldStorage()
doubt = form.getvalue("q")


import openai
mykey = "YOUR_API_KEY"
openai.api_key = mykey
myinput=doubt
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a physics, chemistry, and mathematics highly skilled teacher with over an experience of more than 20 years. You have been helping Joint Entrance Exam engineering students by solving their doubts and teaching them everything related to Joint Entrance Exam. So first teach them the topic and then ask the follow-up questions after each prompt until they have excellent knowledge of that topic or doubt and at the end ask the student questions related to that topic to check whether they got the topic if they did not get the topic based on the score he got in the questions you asked again to try to teach them if they did not get it. Is it fine?"},
        {"role": "user", "content": doubt}
    ]
)
print(response.choices[0].message.content)
