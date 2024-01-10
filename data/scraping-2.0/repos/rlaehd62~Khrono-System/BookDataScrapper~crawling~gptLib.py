from openai import OpenAI
import pandas as pd, json

# OPEN API KEY
KEY = "<API-KEY>"
SYSTEM_PROMPT = """
Extract context-well-explaining keywords from Korean text given, up to 15 in amount in json format.
Extracted Keywords that don't fit into them should be avoided.
Print nothing except json (Format : [ "Keyword", "Keyword", .. "Keyword" ])
Korean Text: \"\"\" 
{}
\"\"\"
"""

client = OpenAI(api_key=KEY)

def extractKeywords(text: str) -> list[str]:
    
    print(f"OpenAPI Request : {text}")
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo-1106",
        messages=[ { 'role': 'system', 'content': f"{SYSTEM_PROMPT.format(text)}" } ],
        temperature=0.5,
        max_tokens=1000
    )
    
    print(response.choices[0].message.content)
    result = json.loads(response.choices[0].message.content)
    return result