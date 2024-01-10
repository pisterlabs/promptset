import asyncio
import json
from langchain.llms import Ollama


async def main():
    ollama = Ollama()
    
    systemprompt = """You will be given a text along with a prompt and a schema. You will have to extract the information requested in the prompt from the
text and generate output in JSON observing the schema provided. If the schema shows a type of integer or number, you must only show a integer for that 
field. A string should always be a valid string. If a value is unknown, leave it empty. Output the JSON with extra spaces to ensure that it pretty 
prints."""
    
    schema = {
        "people": [{
            "name": {
                "type": "string",
                "description": "Name of the person"
            },
            "title": {
                "type": "string",
                "description": "Title of the person"
            }
        }]
    }
    
    with open("./wp.txt", "r") as file:
        textcontent = ' '.join(file.read().split()[:2000])
        
    prompt = f"""Review the source text and determine the 10 most important people to focus on. Then extract the name and title for those people. Output 
should be in JSON.\n\nSchema: \n{json.dumps(schema, indent=2)}\n\nSource Text:\n{textcontent}"""
    
    llm = Ollama(model="mistral:7b",prompt=prompt,systemprompt=systemprompt)
    
    ollama.set_json_format(True)
    await ollama.streaming_generate(prompt, lambda word: print(word, end=''))
    
asyncio.run(main())

