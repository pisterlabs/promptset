from flask import Flask, request, jsonify
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route('/generate_values')
def generate_values():
    message = request.args.get("message")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Given a solution {sol}, give a list of what physical resources it would affect. You do not have to use any of the original resources. The values are either 0, 1, or 2, with 0 creating more of a resource, 1 being neutral in how it affects it and 2 would reduce the certain resource. Try to give more than 4 resources, ideally 16. Try to have some of each type of value. Return the value as a json file with the key being the resource and the value being the value. For example, if the solution is 'reusing more papers and pencils', then the json file would be {'{resource 1}': 1, '{resource 2}': 0, '{resource 3}': 0, '{resource 4}': 2}. No explination needed. Try to find at least some resources that could be negatively affected by the solution, using outside knowledge about what comes from the side effects of a solution. Do not use the example json file, it is just an example."
            },
            {
                "role": "system",
                "content": " Attempt to find both positively and negatively affected resources to the solution. There should be some positive and some negative. You are allowed to use outside resources, such as knowing that making more gas cars can result in more carbon emissions"
            },
            {
                "role": "system",
                "content": "The actual resources have to be physical concrete things, not abstract things like 'happiness' or 'money'. Focus on environmental resources, such as 'water', 'oxygen'. or 'oil'."
            },
            {
                "role": "user",
                "content": message
            },
        ],
        temperature=0.7,
        max_tokens=128
    )

    return str(response.choices[0].message.content)

