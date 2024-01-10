from flask import *
import os
import openai

app = Flask(__name__)
openai.api_key = os.getenv("klucz_api")
@app.route('/')
def index():
    return render_template('./index.html')
@app.route('/obiad', methods=['POST'])
def generate_text():
    skladniki = request.form['skladniki']
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Cześć. Pokaż, prosze przepis na proste,szybkie danie z "+skladniki+"."}
            ]
    )
    #return ({'response': completion.choices[0].message.content})
    response = completion.choices[0].message.content
    return render_template('./obiad.html', response=response)
if __name__ == '__main__':
    app.run()
