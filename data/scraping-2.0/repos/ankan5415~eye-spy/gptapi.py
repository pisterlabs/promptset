import os
import openai
openai.api_key = ("API_KEY")

gpt_role = "I'm a blind person without a cane. You are a camera that has detected several objects in my room and has to tell me what to do next."
user_pre_prompt = "These are the approximations for where the objects in my room are based on what you, as the camera, can see: "
object_locations = "1. Luggage at the bottom of the screen 2. Table on the left 3. Couch on the right. "
user_prompt = "I am moving forward. Tell me what to do next"

def api_call(object_locations, user_prompt):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": gpt_role},
      {"role": "user", "content": f'{user_pre_prompt} {object_locations} {user_prompt}'}
    ]
  )
  
  return print(completion.choices[0].message)

api_call(object_locations, user_prompt)