import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

from utils import sequence_prediction

app = Flask(__name__)
CORS(app)

# load environment variables

load_dotenv()

# initialize OpenAI API and instruction message
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ! TODO: add to dataset date and priority columns! 
instruction_message = {
    "role": "system",
    "content": """
    You are a Finance Assistant who is giving advice through client transaction history. 
    Your role is to help to navigate and manage client finances with ease. 
    Dataset Format is next:
    1. Date - the date of the transaction;
    2. Type - the category of the transaction (e.g., Shopping, Food, Freelance Work);
    3. Amount - the monetary value, with "+" indicating income and "-" indicating expenses.
    Use client transaction history in your response to indicate to a category, where he can save money.
    Tell him an amount of money that he can save in each category.
    Don't forget to tell him how much money he spent in each category.
    In the end, tell him how much money he can save in total.
    All transaction are going to be in the next message.
    """
}


@app.get("/assistant")
def assistant():
    # values = request.json
    messages = [
        {
            "role": "user",
            "content": """
            Client transaction history (json format):
            ["2023-01-01", "Groceries", -45]
            ["2023-01-01", "Utilities", -30]
            ["2023-01-02", "Rent", -500]
            ["2023-01-02", "Freelance Income", 200]
            ["2023-01-03", "Transport", -20]
            ["2023-01-04", "Dining Out", -35]
            ["2023-01-05", "Gym Membership", -25]
            ["2023-01-05", "Salary", 1000]
            ["2023-01-06", "Entertainment", -50]
            ["2023-01-07", "Groceries", -60]
            ["2023-01-08", "Healthcare", -40]
            ["2023-01-09", "Transport", -15]
            ["2023-01-10", "Utilities", -25]
            ["2023-01-11", "Clothing", -75]
            ["2023-01-12", "Freelance Income", 150]
            ["2023-01-13", "Rent", -500]
            ["2023-01-14", "Groceries", -55]
            ["2023-01-15", "Dining Out", -30]
            ["2023-01-16", "Gym Membership", -25]
            ["2023-01-17", "Salary", 1050]
            ["2023-01-18", "Entertainment", -60]
            ["2023-01-19", "Groceries", -70]
            ["2023-01-20", "Healthcare", -45]
            ["2023-01-21", "Transport", -20]
            ["2023-01-22", "Utilities", -30]
            ["2023-01-23", "Clothing", -80]
            ["2023-01-24", "Freelance Income", 250]
            ["2023-01-25", "Rent", -500]
            ["2023-01-26", "Groceries", -65]
            ["2023-01-27", "Dining Out", -40]
            ["2023-01-28", "Gym Membership", -25]
            ["2023-01-29", "Salary", 1100]
            ["2023-01-30", "Entertainment", -55]
            ["2023-01-31", "Groceries", -75]
            ["2023-01-31", "Healthcare", -50]
            ["2023-02-01", "Transport", -25]
            ["2023-02-02", "Utilities", -35]
            ["2023-02-03", "Clothing", -90]
            ["2023-02-04", "Freelance Income", 300]
            ["2023-02-05", "Rent", -550]
            ["2023-02-06", "Groceries", -50]
            ["2023-02-07", "Dining Out", -45]
            ["2023-02-08", "Gym Membership", -30]
            ["2023-02-09", "Salary", 1150]
            ["2023-02-10", "Entertainment", -70]
            ["2023-02-11", "Groceries", -80]
            ["2023-02-12", "Healthcare", -60]
            ["2023-02-13", "Transport", -30]
            ["2023-02-14", "Utilities", -40]
            ["2023-02-15", "Clothing", -100]
            # Client question: How can I save more money?"""
        }
    ]

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[instruction_message, *messages],
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

@app.get("/forecast")
def forecast():
    return jsonify(sequence_prediction())

@app.get("/imagetotext")
def imagetotext():
    return jsonify(sequence_prediction())

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=os.getenv("FLASK_PORT", 5000), 
        debug=False
    )
