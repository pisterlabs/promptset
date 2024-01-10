import openai
from flask import request, Flask, flash, redirect


app = Flask(__name__)
openai.api_key = "yourAPI_KEY"


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.route("/translate", methods=['POST'])
def translate_text():
    if request.method == 'POST':
        text = request.json.get('text')
        target_language = request.json.get('dest_language')
        if type(text) == str:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=f"Translate the following text into {target_language}: {text}\n",
                max_tokens=60,
                n=1,
                stop=None,
                temperature=0.9, )
            return response.choices[0].text.strip(), 200
        elif type(text) == list:
            result = []
            for t in text:
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=f"Translate the following text into {target_language}: {t}\n",
                    max_tokens=60,
                    n=1,
                    stop=None,
                    temperature=0.9, )
                result.append(response.choices[0].text.strip())
            return result, 200
        else:
            return "Wrong type of variable. Expect key name 'text' is list or str", 400
    return "Wrong type of request. Expect POST request", 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
