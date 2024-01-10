from openai import OpenAI
import os
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS, cross_origin
from supabase import create_client

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

load_dotenv("../config/.env")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

supabase_client = create_client(url, key)

response = supabase_client.from_("Product").select("*").execute()
data = response.data
names = []

for d in data:
    names.append(d["name"])

ingredients = ",".join(names)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def query(input):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"You are a shopping assistant. You will give a csv list of ingredients when the user prompts you with a food, only using this list of ingredients: {ingredients}. Do not provide an additional input or help. Just ingredients, and try to keep the ingredients as basic as possible. List as many as you can. If the input is not a valid food, return ''. Also, only characters permitted are commas and lowercase letters. No other punctuation! No parenthesis or periods!",
            },
            {
                "role": "user",
                "content": f"What are the ingredients I need to make {input}?",
            },
        ],
    )

    answer = (completion.choices[0].message.content).split(",")
    cleaned_answer = [item.strip() for item in answer]
    return cleaned_answer


@app.route("/query/<food>")
@cross_origin()
def get_ingredients(food):
    return query(food), 200


@app.route("/")
@cross_origin()
def check_active():
    return "OK", 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
