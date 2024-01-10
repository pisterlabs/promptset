import os
from openai import OpenAI
from dotenv import load_dotenv
import json


def openai_request(prompts, model_engine='gpt-3.5-turbo'):
    client = OpenAI(
        # This is the default and can be omitted
        api_key = os.environ.get("OPENAI_API_KEY")
    )

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={ "type": "json_object" },
        messages= [ {"role": "system", "content": "You are a helpful assistant designed to output JSON."}] + prompts
    )

    json_response = response.choices[0].message.content

    return json_response

def get_new_line(history, players, additional_prompt):
    new_line_json = openai_request(
        prompts = [
            {"role": "user", "content": "We are in an improv show. Here are the lines so far: " + history},
            {"role": "user", "content": "These are the players: " + players},
            {"role": "user", "content": "Generate the new line for the scene and return it as JSON. I want the new line to contribute to the story development. There should be two fields: 'player' and 'line'."},
        ]
    )
    player = json.loads(new_line_json)['player']
    line = json.loads(new_line_json)['line']
    return player, line

def get_players(num_players):
    """
    Returns a list of players in the scene.
    """
    players_json=  openai_request(
        prompts = [
            {"role": "user", "content": "I am starting a improv scene. Give me names of " + str(num_players) + " players in the scene. Return the the names in a list in JSON with the field 'players'."},
        ]
    )
    return json.loads(players_json)['players']

def get_first_line(prompt, players):
    """
    Returns the first line of the scene.
    """
    first_line_json = openai_request(
        prompts = [
            {"role": "user", "content": "I am starting a improv scene. Here is the prompt I want you to kickstart a scene: " + prompt},
            {"role": "user", "content": "These are the players: " + " ".join(players)},
            {"role": "user", "content": "Generate the first line of the scene and return it as JSON. There should be two fields: 'player' and 'line'."},
        ]
    )
    line = json.loads(first_line_json)['line']
    player = json.loads(first_line_json)['player']
    return player, line

def print_intro():
    print("")
    print("**********")
    print("Welcome to the improv scene generator!")
    print("**********")
    print("")

def print_enjoy_the_show():
    print("")
    print("**********")
    print("Enjoy the show!")
    print("**********")
    print("")

def main():
    print_intro()
    num_players = int(input("Enter the number of players: "))
    prompt = input("Enter the prompt for the scene: ")
    history = []
    players = []
    conversation_rounds = 10
    print_enjoy_the_show()
    for i in range(conversation_rounds):
        if i == 0:
            players = get_players(num_players)
            player, line = get_first_line(prompt, players)
            history.append(line)
            print(player + ": " + line)
            print("")
        else:
            player, line = get_new_line(" ".join(history), " ".join(players), "")
            history.append(player + ": " + line)
            print(player + ": " + line)
            print("")
    return


load_dotenv() 
main()