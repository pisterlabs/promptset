from dotenv import load_dotenv
import os
from flask import Flask, request
from openai import OpenAI

load_dotenv()

# print(os.environ.get("TEST"))

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# engines = openai.Engine.list()
# completion = openai.Completion.create(engine="text-davinci-003", prompt="This is a test", max_tokens=5)
# print(completion)

app = Flask(__name__)

@app.route("/nlp", methods=["POST"])
def nlp():
    if(request.form.get("prompt") == None):
        return {"error": "No prompt provided"}
    
    try:
        prompt = request.form.get("prompt")

        max_tokens = request.form.get("max_tokens")
        if(max_tokens == None):
            max_tokens = 50
        else:
            max_tokens = int(max_tokens)
        max_tokens = max_tokens if max_tokens <= 150 else 150
                
        temp = request.form.get("temp")
        if(temp == None):
            temp = 0.5
        else:
            temp = float(temp)
        temp = temp if temp >= 0 and temp <= 1 else 0.5

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-3.5-turbo",
            max_tokens=max_tokens,
            temperature=temp,
        )

        # completion = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_tokens, temperature=temp)
        
        print(chat_completion)

        return chat_completion.choices[0].message.content
    except Exception as e:
        return {"error": "The request was not successful. Check your parameters."}

if __name__ == "__main__":
    app.run(debug=True)