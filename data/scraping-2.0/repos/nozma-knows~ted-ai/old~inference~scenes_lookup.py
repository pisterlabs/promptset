import json
from typing import List
from marvin import ai_model
from marvin import openai
from pydantic import BaseModel
from summary_get import Shot
from summary_get import Scene

class Scene(BaseModel):
    '''Take into account the movie and the context when generating the scenes. Provide additional detail when possible.'''
    movie_name: str
    shot_list:List[Shot]
    
data = json.load(open('/Users/ianlaffey/ted-ai/inference/scenes.json', 'r'))

scenes = Scene.model_validate_json(data)

print(scenes)

system_prompt = f'''Given a visual description of the first 5 minutes of Ferris Buller's day off, please create mappings from the visual descriptions of characters to the characters names. Example: "Man in Blue shirt -> John Smith"
'''

messages = [
    {"role": "system", "content" : system_prompt},
    {"role": "user", "content" : str(scenes)},
]
res = openai.ChatCompletion('gpt-4').create(messages = messages).response
# print(res)
print(res.choices[0].message.content)

# write to file

with open('characters.json', "w") as file:
    json.dump(res.choices[0].message.content, file, indent=4)