import os
import openai
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
openai.api_key = os.environ["OPENAI_API_KEY"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/rate_code', methods=['POST'])
def rate_code():
    code_snippet = request.json.get('code_snippet')
    
    if not code_snippet:
        return jsonify({"error": "No code snippet provided"}), 400

    prompt = f"""
Evaluate the following code snippet based on:

1. Correctness (0-40): Does the code run without errors and produce the expected output?
2. Readability (0-30): Is the code easy to read and understand?
3. Algorithm (0-30): Is the algorithm used efficient and well-implemented?

Code snippet:
{code_snippet}

Correctness:
Readability:
Algorithm:
"""
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    response_text = response.choices[0].text.strip()
    try:
        correctness_str, readability_str, algorithm_str = response_text.split("\n", 2)
        correctness = int(correctness_str.split(":")[-1].strip())
        readability = int(readability_str.split(":")[-1].strip())
        algorithm = int(algorithm_str.split(":")[-1].strip())

        total_rating = correctness + readability + algorithm
        return jsonify({
            "correctness": correctness,
            "readability": readability,
            "algorithm": algorithm,
            "total_rating": total_rating,
        })
    except ValueError:
        return jsonify({"error": "Failed to parse ratings from AI response"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
