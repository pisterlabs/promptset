from flask import jsonify, Blueprint, request
import openai

character_blueprint = Blueprint("character", __name__)

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",  # Switch to GPT-3.5-turbo
        prompt=prompt,
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

@character_blueprint.route('/create_random_character', methods=['POST'])
def create_random_character():
    prompt = (
        "Create a D&D 5e character in the following format:\n\n"
        "Name: [Character Name]\n"
        "Race: [Race]\n"
        "Subrace: [Subrace if applicable]\n"
        "Class: [Class]\n"
        "Background: [Background]\n"
        "Backstory: [Detailed backstory including the character's origin and motivations "
        "for being an adventurer.]\n\n"
        "Ability Scores:\n"
        "Strength (STR): [Strength score]\n"
        "Dexterity (DEX): [Dexterity score]\n"
        "Constitution (CON): [Constitution score]\n"
        "Intelligence (INT): [Intelligence score]\n"
        "Wisdom (WIS): [Wisdom score]\n"
        "Charisma (CHA): [Charisma score]\n\n"        
    )

    character_description_and_stats = generate_response(prompt)
    return jsonify({"character_description_and_stats": character_description_and_stats})

@character_blueprint.route('/create_custom_character', methods=['POST'])
def create_custom_character():
    name = request.json.get("name", "a character")
    character_class = request.json.get("class", "random class")
    race = request.json.get("race", "random race")
    subrace = request.json.get("subrace", "")

    if subrace:
        race = f"{subrace} {race}"

    prompt = (
        f"Create a D&D 5e {race} {character_class} character named {name} with a detailed "
        "backstory that includes their origin and motivations for being an adventurer, and "
        "their ability scores (Strength, Dexterity, Constitution, Intelligence, Wisdom, and "
        "Charisma). For races with subraces, always include a subrace. Ensure the character's name, race, "
        "and class, and all stats are clearly mentioned in the response."
    )
    character_description_and_stats = generate_response(prompt)
    return jsonify({"character_description_and_stats": character_description_and_stats})