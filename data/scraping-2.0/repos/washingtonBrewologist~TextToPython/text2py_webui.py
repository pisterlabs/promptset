import openai
from flask import Flask, request, render_template
from markdown import markdown

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        api_key = request.form.get('api_key')
        tasks = request.form.get('tasks').split(',')

        openai.api_key = api_key

        task_str = "\n".join([f"{idx+1}. {task.strip()}" for idx, task in enumerate(tasks)])
        prompt = f"\"\"\"\nYour task is to write Python code to: \n{task_str}\n\"\"\""

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates plain English into Python code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        result = completion.choices[0].message['content']

        # Pass result as a variable to the template
        return render_template('home.html', result=result)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
