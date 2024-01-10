"""
This scratch file explores the idea of having LLMs specifically generate content to help with learning APIs
"""
import json

import openai

from app import babotree_utils
from app.database import get_direct_db
from app.models import Highlight, HighlightSource

together_openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('TOGETHER_API_KEY'),
    base_url="https://api.together.xyz/v1",
)

openai_openai_client = openai.OpenAI(
    api_key=babotree_utils.get_secret('OPENAI_API_KEY'),
)

openai_client = together_openai_client
# model = 'gpt-4-1106-preview'
model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'


def extract_json(content):
    """
    We try to find a point in the text that is likely to contain a json object, and then extract it
    :param content:
    :return:
    """
    if "```json" in content:
        json_start = content.find("```json\n")
        json_end = content.find("```", json_start + 1)
        json_str = content[json_start + 7:json_end]
        print(json_str)
        return json_str.strip()
    else:
        # look for first { and last }
        json_start = content.find("{")
        json_end = content.rfind("}")
        json_str = content[json_start:json_end + 1]
        print(json_str)
        return json_str.strip()


def main():
    db = get_direct_db()
    highlight_source_id = '5269a1c0-1774-4f25-ab7c-a41554c2a0b4'
    highlight_source = db.query(HighlightSource).filter(HighlightSource.id == highlight_source_id).first()
    docker_highlights = db.query(Highlight).filter(Highlight.source_id == highlight_source_id).all()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"Here are some excerpts from a source text called \"{highlight_source.readable_title}\" :\n" + "\n".join(
                [highlight.text for highlight in docker_highlights])
        },
        {
            "role": "assistant",
            "content": "Ok, I'm ready."
        },
        {
            "role": "user",
            "content": "What is the topic of the excerpts?"
        }
    ]
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    print("-----")
    # append the response message and ask more questions
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    messages.append({
        "role": "user",
        "content": "Nice. What are the key APIs in this subject? Provide a list containing the names of 5 key API classes or methods and their definition/use. Be concise."
    })
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    print("-----")
    # append the response message and ask more questions
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    messages.append({
            "role": "user",
            "content": "Please convert this list of term/definition pairs to a json object obeys this TypeScript interface definition:\n\n" +
            "interface ApiDefinition {\nname: string;\ndefinition: string;\n}"
            "interface ApiDefinitions {\nresults: ApiDefinition[];\n}"
        })
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=.25,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=.25,
        presence_penalty=.25,
    )
    print(response.choices[0].message.content)
    print("-----")
    # now let's try to extract/parse the json from the text
    try:
        json_str = extract_json(response.choices[0].message.content)
        print("Found this json object:")
        print(json_str)
        json_dict = json.loads(json_str)
        print("Parsed json:")
        print(json_dict)
    except Exception as e:
        print(e)




if __name__ == '__main__':
    main()