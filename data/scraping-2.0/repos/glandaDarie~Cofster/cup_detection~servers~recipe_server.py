from typing import Dict, Literal
from flask import Flask, Response, jsonify, request
import sys
from typing import Tuple

# Temporary workaround: Adding the parent directory to the sys.path
# This is necessary for relative imports to work in the current project structure.
# Please consider restructuring the project to eliminate the need for this workaround.
sys.path.append("../")

from utils.constants import PROMPT_TEMPLATE
from services.llm_services.openAIService import OpenAIService
from utils.paths import PATH_COFFEE_CREATION

app = Flask(__name__)

@app.route("/coffee_recipe", methods=["GET", "PUT"])
def coffee_recipe() -> (tuple[Response, Literal[200]] | None):
    if request.method == "GET":
        return get_coffee_recipe()
    elif request.method == "PUT":
        return put_coffee_recipe()
    else:
        return "Method Not Allowed", 405

def get_coffee_recipe() -> Tuple[Response, int]:
    coffee_name : str = request.args.get("coffee_name")
    if not coffee_name:
        return jsonify({"error", "Coffee name was not provided"}), 400
    try:
        openai_service : OpenAIService = OpenAIService()
        prompt_recipe : str = PROMPT_TEMPLATE.format(coffee_name)
        coffee_ingredients : Dict[str, str] = openai_service(prompt=prompt_recipe)
        response : Dict[str, str] = {"ingredients": coffee_ingredients}
    except Exception as exception:
        return jsonify({"error" : f"Server side error: {exception}"}), 500
    return jsonify(response), 200

def put_coffee_recipe() -> Tuple[Response, int]:
    #TODO
    pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
    