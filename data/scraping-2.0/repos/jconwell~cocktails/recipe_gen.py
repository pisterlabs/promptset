import time
from pathlib import Path
from dotenv import dotenv_values
from openai import OpenAI

system_prompt = f"""You are an expert bartender with years of experience.
You have made thousands of cocktails and know the recipes for all of them.
When I ask you for a cocktail recipe you will reply with the recipe for a cocktail that I ask for in json format.
Include the following fields in the json structure: 
- a string field for the name of the ingredient.
- a numeric field for the volume of ingredient, measured in in ounces if liquid and count if solid.
- a string field for the unit of measure of the ingredient.
- a boolean field where true indicates the ingredient is a liquid and false indicates it is a solid.
- a float field, from 0 to 1, that indicates how popular you think the cocktail is.
- a float field, from 0 to 1, that indicates how difficult you think the cocktail is to make.
- a json array with step by step instructions on how to make the cocktail.
- a json array for the taste profile of the cocktail.
- a string field for any garnishes that should be added to the cocktail.
- a float field for the alcohol content of the cocktail.
I will be asking for the recipe for thousands of cocktails and the returned json structure will be used to create features for a machine learning model. 
Think through what other cocktail recipe characteristics would be useful for a machine learning model and include any fields you think would be useful in the json document.
Only return the json structure for the cocktail that I ask for, and no commentary or other text 
"""

user_prompt = """
Return the recipe for the cocktail delimited by single quotes.

'{}'
"""

# load environment variables
config = dotenv_values(".env")


def _call_gpt(client, cocktail_name):
    header = "Authorization: Bearer OPENAI_API_KEY"
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt.format(cocktail_name)
            }
        ],
        # stream=True,
    )
    # for chunk in stream:
    #     if chunk.choices[0].delta.content is not None:
    #         print(chunk.choices[0].delta.content, end="")
    cocktail_json = stream.choices[0].message.content
    return cocktail_json


def call_gpt(client, cocktail_name, attempts=3, backoff=2):
    for attempt in range(attempts):
        try:
            return _call_gpt(client, cocktail_name)
        except Exception as e:
            print(f"Error calling GPT: {e}")
            time.sleep(backoff * attempt)


def get_recipes(client, in_path, out_path):
    with open(in_path, 'r') as handle:
        for i, cocktail_name in enumerate(handle.readlines()):
            cocktail_name = cocktail_name.strip()
            cocktail_path = out_path + cocktail_name.replace(" ", "_") + ".json"
            # remove chars from file name
            for char in ['(', ')', "'", '"', ',', '&']:
                cocktail_path = cocktail_path.replace(char, '')
            if not Path(cocktail_path).exists():
                # go get the recipe
                cocktail_json = call_gpt(client, cocktail_name)
                # write recipe to file
                with open(cocktail_path, 'w') as out_handle:
                    out_handle.write(cocktail_json)
                print(f"Got recipe for {cocktail_name} ({i})")


def main():
    client = OpenAI(
        api_key=config['API_KEY'],
        organization=config['ORG_KEY'],
    )
    in_path = "cocktails.txt"
    out_path = "recipes/"
    get_recipes(client, in_path, out_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()