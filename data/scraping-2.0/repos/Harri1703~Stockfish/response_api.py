import os
from flask import Flask, request, jsonify

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

agent = create_csv_agent(OpenAI(temperature=0), "kochi.csv", verbose=False)

app = Flask(__name__)


@app.route("/new-question", methods=["POST"])
def new_question():
    data = request.get_json()
    ans = agent.run(data["question"])
    response = {"answer": ans}

    text_to_write = f"User : {data['question']}\nMPEDA Assistant : {ans}\n"

    with open("transcript.txt", "a") as file:
        file.write(text_to_write + "\n")

    return jsonify(response), 201


if __name__ == "__main__":
    app.run(debug=True, port=7002)
