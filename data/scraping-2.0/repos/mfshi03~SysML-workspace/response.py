import os
import openai 
import re
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

pattern = re.compile(r"\`\`\`sysml([\s\S]*?)\`\`\`")

prompts = [
    "Generate SysMLv2 code for {}",
    "Convert the following SysMLv2 code to python: {}",
]

def completion(input:str, i:int=0) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompts[0].format(input)
            },
        ],
        temperature=0,
        max_tokens=500,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0]["message"]["content"]

text = completion("Rocket ship")
print(re.findall(pattern, text)[0])


