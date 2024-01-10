import openai
from flask import Blueprint, jsonify, request
from main import app, db
from main.models import AdventureLocations, AdventureNPCs, Adventures
from main.util import utilities
from main.util.api_key import API_KEY


routes = Blueprint("routes", __name__)

openai.api_key = API_KEY


@routes.route(
    "/generate_adventure", methods=["POST"]
)  ## occasional error: raise JSONDecodeError("Expecting value", s, err.value) from None json.decoder.JSONDecodeError: Expecting value: line 1 column 21 (char 20)
def generate_adventure():
    """
    create adventure based on user input using GPT-3 and return the result
    """

    message_body = request.json

    prompt = f'You are a professional writer of RPG Adventures who is tasked with\
    creating an adventure with the title {message_body["adventureTitle"]}. The adventure is supposed to be\
    set in a {message_body["adventureSetting"]} setting. The general plot of the adventure should be based\
    on the following: {message_body["adventurePlot"]}.\
    Please write a detailed adventure that includes a breakdown of\
    the following:\
    1. The adventure hook\
    2. The adventure plot\
    3. The adventure climax\
    4. The adventure resolution\
    5. Important NPCs and monsters\
    Please use the above structure and use a minimum of 1000 words for your answer.\
    Format the answer as a json object with the following stucture:\
    {{  "AdventureTitle": content,\
        "AdventureHook": content,\
        "AdventurePlot": content,\
        "AdventureClimax": content,\
        "AdventureResolution": content,\
        "AdventureNPCs": content}}'
    response = openai.Completion.create(
        engine="text-davinci-003", prompt=prompt, max_tokens=2000
    )
    expected_keys = [
        "AdventureTitle",
        "AdventureHook",
        "AdventurePlot",
        "AdventureClimax",
        "AdventureResolution",
        "AdventureNPCs",
    ]
    gpt_json = utilities.clean_gpt_response(
        response["choices"][0]["text"], expected_keys
    )

    # create adventure object for new adventure
    adventure = Adventures(
        adventure_title=gpt_json["AdventureTitle"],
        adventure_hook=gpt_json["AdventureHook"],
        adventure_plot=gpt_json["AdventurePlot"],
        adventure_climax=gpt_json["AdventureClimax"],
        adventure_resolution=gpt_json["AdventureResolution"],
        adventure_npcs=gpt_json["AdventureNPCs"],
    )

    db.session.add(adventure)
    db.session.commit()

    # return json response
    response = jsonify({"status": "success", "message": gpt_json})
    response.status_code = 201
    return response


@routes.route("/get_adventures_from_db", methods=["GET"])
def get_adventures_from_db():
    """
    get all or single adventure(s) from database
    """

    if request.args.get("id"):
        adventure_id = request.args.get("id")
        adventures = Adventures.get_adventures(adventure_id)
    else:
        adventures = Adventures.get_adventures()

    response = jsonify(adventures)

    # reduce to required fields
    response.status_code = 200
    return response


@routes.route("/delete_adventures_from_db", methods=["DELETE"])
def delete_adventures_from_db():
    """
    delete adventure(s) from database
    """

    if request.json:
        adventure_ids = request.json["ids"]
        Adventures.delete_adventures(adventure_ids)

    response = jsonify({"status": "success", "message": "adventure deleted"})
    response.status_code = 204
    return response


@routes.route("/extract_entities/<id>", methods=["POST"])
def extract_entities(id):

    adventure = Adventures.get_adventures(id)[0]

    npc_list, locations_list = utilities.extract_entities_from_adventure(adventure)

    response = jsonify({"status": "success", "message": [npc_list, locations_list]})
    response.status_code = 201

    return response


@routes.route("/get_NPCs_from_db", methods=["GET"])
def get_NPCs_from_db():
    """
    get NPCs from database
    """

    adventure_id = request.args.get("id")
    npc = AdventureNPCs.get_NPCs(adventure_id)

    response = jsonify(npc)

    # reduce to required fields
    response.status_code = 200
    return response


@routes.route("/get_locations_from_db", methods=["GET"])
def get_locations_from_db():
    """
    get locations from database
    """

    adventure_id = request.args.get("id")

    npc = AdventureLocations.get_locations(adventure_id)

    response = jsonify(npc)

    # reduce to required fields
    response.status_code = 200
    return response


@routes.route("/generate_npc", methods=["POST"])
def generate_npc():
    """Create a new NPC for an Adventure using GPT-3 and return the result"""

    if request.json["adventureId"] == "":
        return "Please provide adventure id", 400

    npc = {}
    npc["name"] = request.json["characterName"]
    npc["game_system"] = request.json["selectedSystem"]
    npc["adventure_id"] = request.json["adventureId"]
    npc["game_system_version"] = request.json["selectedSystemVersion"]

    if npc["game_system"] == "Define other":
        npc["game_system"] = request.json["custom_system"]

    adventure = Adventures.get_adventures(npc["adventure_id"])[0]

    prompt = f'You are a professional writer of RPG Adventures who is tasked with\
    creating an background and game stats for Non Player Characters. \
    The NPC is called {npc["name"]} and is supposed to be\
    set in a {npc["game_system"]} {npc["game_system_version"]} setting. The background of the NPC should be compatible with\
    the following adventure: {adventure["AdventureHook"]} . {adventure["AdventurePlot"]}.\
    {adventure["AdventureClimax"]}.{adventure["AdventureResolution"]}.\
    Please use the above structure and use a minimum of 1000 words for your answer.\
    Format the answer as a json object with the following stucture. The NPCStats should use the common Statblock format of the respective game system:\
    {{  "NPCBackground": content,\
        "NPCStats": content,\
        }}'
    gpt_response = openai.Completion.create(
        engine="text-davinci-003", prompt=prompt, max_tokens=3000
    )
    gpt_response_text = gpt_response["choices"][0]["text"]

    gpt_response_text = utilities.clean_gpt_response(
        gpt_response_text, expected_keys=["NPCBackground", "NPCStats"]
    )

    npc.update(gpt_response_text)
    npc = AdventureNPCs(
        npc_name=npc["name"],
        npc_background=npc["NPCBackground"],
        npc_stats=npc["NPCStats"],
        adventure_id=npc["adventure_id"],
        npc_game_system=npc["game_system"],
    )

    if not app.config["TESTING"]:
        db.session.add(npc)
        db.session.commit()

    response = jsonify({"status": "success", "message": gpt_response_text})
    response.status_code = 201
    return response
