from flask import Flask, request, jsonify
import openai
import os
from values import get_idea_list

# To Be Retrieved from database
ideas = ["Uber", "...","..."]
resources = ["water", "metal", "gas", "paper"]
values = [[1, 0, 2, 1], [2, 2, 0, 1], [0, 1, 1, 2]]

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route('/generate_complement')
def generate_complement(ideaid):
    # Find idea with most complements
    idea = values[ideaid]
    max_index = ideaid
    max_list = idea
    length = len(values[0])
    max_count = 0

    for index, ideavals in enumerate(values):
        count = 0
        for i in range(length):
            if (idea[i] == 2 and ideavals[i] == 0) or (idea[i] == 0 and ideavals[i] == 2):
                count += 1
        if count > max_count:
            max_count = count
            max_index = index
            max_list = ideavals

    complementary_materials = []
    for i in range(length):
        if (idea[i] == 2 and max_list[i] == 0) or (idea[i] == 0 and max_list[i] == 2):
            complementary_materials.append(resources[i])

    # Get that idea
    given_idea = ideas[ideaid]
    compatible_idea = ideas[max_index]

    #OPENAI STUFF HAVEN"T TESTED YET
    message = request.args.get("message")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Explain how the following two solutions to sustainability problems complement each other: Solution 1: "{given_idea}", Solution 2: "{compatible_idea}". For example, one solution could use up the excess materials created in the other solution."
            },
            {
                "role": "user",
                "content": message
            },
        ],
        temperature=0.7,
        max_tokens=128
    )

    return compatible_idea, complementary_materials, str(response.choices[0].message.content)



