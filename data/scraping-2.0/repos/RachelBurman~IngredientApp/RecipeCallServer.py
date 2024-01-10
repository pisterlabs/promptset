#server.py
from flask import Flask, jsonify
import OpenAIRecipeGeneration as oarg
import OpenAIattempt2 as oar
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/generate_random_recipe')
def generate_random_recipe():
    recipe = oarg.generate_random_recipe()
    return jsonify(recipe)

@app.route('/generate_random_pairings')  # Add a route for the generate_random_pairings function
def generate_random_pairings():
    pairings = oar.generate_random_pairings()
    return jsonify(pairings)  # Return the response to the client

if __name__ == '__main__':
    app.run(debug=True)