import json
from flask import redirect, request, url_for, render_template, flash
from flask.views import MethodView
from openai import OpenAI


class Poem(MethodView):
    def get(self):
        image_path = request.args.get("image_path")
        labels = request.args.get("labels")
        objects = request.args.get("objects")
        poem = self.createPoem(labels, objects)
        if "error" in poem:
            title = "Error"
            poem = poem["error"]
        else:
            title = poem["title"]
            poem = poem["poem"]

        return render_template(
            "poem.html",
            image_path=image_path,
            title=title,
            poem=poem,
            labels=labels,
            objects=objects,
        )

    def createPoem(self, labels, objects):
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.createSystemRole()},
                {"role": "user", "content": self.createUserPrompt(labels, objects)},
            ],
        )
        poem = completion.choices[0].message.content
        try:
            poem_json = json.loads(poem)
            return poem_json
        except ValueError as e:
            return {
                "error": "Hmm, something went wrong. Please try to 'Regenerate Poem'."
            }

    def createUserPrompt(self, labels, objects):
        prompt = f"Create a poem about {labels} and {objects}."
        return prompt

    def createSystemRole(self):
        role_prompt = 'You are a master artist of poetry who is skilled in creating poems of all styles. When you answer the user\'s prompt, please output in json in the following format: {"title": <title>, "poem": <poem>} make sure to use double quotes for the json keys and values. Also you must make sure to escape any newline characters in the poem with a backslash. The json output must be compatible with the json.loads() function in python.'
        return role_prompt
