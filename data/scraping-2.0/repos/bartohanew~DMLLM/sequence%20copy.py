from common import *
import json
from pymongo import MongoClient
client = MongoClient()

storyline = list(client.lost_mines.story.find())

story_part_name = 'start'

from modularity import OpenAI

client = OpenAI()

# def summarize_plotline(messages):
#     message_text = "\n".join([f"+ {x['role']}: {x['content']}" for x in messages])
#     prompt = f"""
#     Your goal is to summarize the plotpoints contained in the following conversation between a DM and a player.
#     In each plot point, be as specific as possible.
#     Keep note of any characters, locations, or items that are mentioned.
#     Do not include any information not present in the following messages!

#     Messages:
#     {message_text}
#     """
#     print(prompt)

#     messages = [
#         {"role": "system", "content": prompt},
#     ]

#     response = get_response(messages)#, model="gpt-3.5-turbo")
#     print('Summarized!')
#     print(response)
#     return response


M = []
summary = []

def respond():
    global M, summary, story_part_name, storyline

    story_part = [x for x in storyline if x['name'] == story_part_name][0]

    next_steps = "\n".join([f"\t{x}: {y}" for x, y in story_part['next_steps'].items()])

    my_messages = []

    prompt = f"""
You are the Dungeon Master (DM), using the DnD 5E rules and content. Be clever, witty, and sometimes funny. But serious when the time comes
I am the player.
You are not the player.
I am not the DM.
You are currently in the scene "{story_part['name']}"
The player's character is in a four person party
When the player creates a character, you take on the role of the remaining three members of the player's party
Roleplay these characters. Give them personalities. Invent backgrounds for them that tie them to the world, but do not state this information to the player. Take actions you think they would make. 

Be sure to have the make skill checks throughout the adventure when it seems appropriate. But do not state the numbers you roll unless asked. It ruins the immersion. You roll for the player and state the outcome.

When combat starts, consult the 5E rules for combat.

During combat, you take the turn of the NPCs. Play out their turns in the initiative order and do not move on to the next character in initiative until you have completed the current character's turn

If you want to change the scene, type:
{{"type": "change_scene", "to": "scene name"}}



Description of the current scene:
    {story_part['description']}

Scenes available, their names and descriptions:
{next_steps}


    """
    #Otherwise, any (non-JSON) message you type will be sent to the player. (I REMOVED THIS TO TRY TO DEAL WITH THE RAMBLING MESSAGES)
    #print(prompt)
    my_messages.append({'role': 'system', 'content': prompt})

    response = get_response(my_messages + M)#, model="gpt-3.5-turbo")

    # determine if the response is an action
    is_action = False
    try:
        response = json.loads(response)
        is_action = response['type'] == 'change_scene'
    except:
        pass

    # if not, just add it to the messages
    if not is_action:
        M.append({"role": "assistant", "content": response})
        print("\n\n" + "DM: " + response + "\n\n")

    # if so, change the scene
    else:
        story_part_name = response['to']
        print(f"Changed to scene: {story_part_name}")

        M.append({"role": "system", "content": f"Changed to scene: {story_part_name}"})

        # since the computer used its turn on changing the scene, let it go again (to respond with text)
        respond()

    # consolidate things, if it's getting long
    # if len(M) > 10:
    #     # remember a summary of the messages
    #     summary.append(summarize_plotline(M))

    #     # clear the messages
    #     M = M[-2:]




# this is actually the interactive loop!
while True:
    # human does its thing
    query = input("Player: ")
    M.append({"role": "user", "content": query})

    # computer does its thing
    respond()