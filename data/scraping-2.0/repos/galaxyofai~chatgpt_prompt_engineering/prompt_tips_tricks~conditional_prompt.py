

import openai
import os 
from dotenv import load_dotenv,find_dotenv
__ = load_dotenv(find_dotenv()) #read local .env file
openai.api_key=os.environ["OPENAI_API_KEY"]

def get_completion(prompt,model="gpt-3.5-turbo"):
    messages=[{"role":"user","content":prompt}]
    response=openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message['content']



text=f"""
Learning AI is easy! First go to the site www.galaxyofai.com,read some articles related to AI,
try to understand it, try to implements it for your projects,
And that's it! Now you have idea of how we can implements AI in real life.
Enjoy the learning AI.

"""

text_2=f"""
Prompt engineering is the process of creating clear and concise instructions, known as prompts, 
that help large language models generate text, translate languages, 
write creative content, and answer questions in an informative way.

"""
prompt_example=f"""
You will be provided with the text delimited by tripple quotes.
If it contain a sequence of instructions ,
re-write those instructions in the following format:

Step 1 - ...
Step 2 - ...
...
Step N - ...

If the text does not contain a sequence of instructions,
then simply write "No steps provided"
```{text_2}```
"""
response=get_completion(prompt=prompt_example)
print(response)