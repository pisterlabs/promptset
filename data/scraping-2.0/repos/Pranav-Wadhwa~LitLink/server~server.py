import constants
from langchain import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from models import CharacterList, RelationshipDescription, Relationship, Character
from flask import Flask, request
from flask_cors import CORS, cross_origin
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

cache = dict() # {act_number: object}

with open('cache.json', 'r') as cachefile:
    cache = json.loads(cachefile.read())

model_name = "gpt-4"
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature, openai_api_key=constants.OPENAI_API_KEY)

def get_relationships(character1: str, character2: str, act_number: int) -> Relationship|None:
    print(f'Checking relationship between {character1} and {character2}')
    query = f"Is there a significant relationship between {character1} and {character2} in Act {act_number}? If so, describe it."
    parser = PydanticOutputParser(pydantic_object=RelationshipDescription)
    template = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt = template.format_prompt(query=query)
    output = model(prompt.to_string())
    relationship: RelationshipDescription = parser.parse(output)
    print('is significant: ', relationship.is_significant)
    if relationship.is_significant:
        return Relationship(character1, character2, relationship.description)
    return None

@app.route('/get_characters')
@cross_origin()
def get_characters():
    act_number = request.args.get('act_number')
    if act_number in cache:
        return cache[act_number]
    print('Getting all characters')
    query = "Who are all of the characters in Romeo and Juliet Act {}? Return their full names".format(act_number)
    parser = PydanticOutputParser(pydantic_object=CharacterList)
    template = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt = template.format_prompt(query=query)

    characterList: CharacterList = parser.parse(model(prompt.to_string()))

    print(characterList)

    relationships: list[Relationship] = []

    for i, character1 in enumerate(characterList.characters):
        for character2 in characterList.characters[i + 1:]:
            relationship = get_relationships(character1.name, character2.name, act_number)
            if relationship:
                relationships.append(relationship)

    obj = {
        "characters": list(map(lambda x: x.toJSON(), characterList.characters)), 
        "relationships": list(map(lambda x: x.toJSON(), relationships))
    }
    cache[act_number] = obj
    print(obj)
    with open('cache.json', 'w') as cachefile:
        cachefile.write(json.dumps(obj))
    return obj

if __name__ == '__main__':
    app.run()