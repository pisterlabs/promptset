from common import *
import json

storyline = [
    {
        'name': "start",
        'description': "In Neverwinter, Gundren Rockseeker, a dwarf, hires you to transport provisions to Phandalin. Gundren, with a secretive demeanor, speaks of a significant discovery he and his brothers made. He promises ten gold pieces for safe delivery to Barthen's Provisions in Phandalin. Accompanied by Sildar Haliwinter, he leaves ahead on horseback. Your journey begins on the High Road from Neverwinter, moving southeast. Danger lurks on this path, known for bandits and outlaws.",
        'next_steps': {
            'introduce characters': "A good first step",
            'determine marching order': "Optional",
            'driving the wagon': "When the journey really begins. Make sure they know some of the plot before beginning."
        }
    },
    {
        'name': "introduce characters",
        'description': "Players take turns introducing their characters. They describe their appearance, background, and how they came to know Gundren Rockseeker. Encourage creativity in their backstories, like childhood friendships or past rescues by Gundren.",
        'next_steps': {
            'determine marching order': "At any time."
        }
    },
    {
        'name': "determine marching order",
        'description': "The party discusses and decides their traveling formation. Who will drive the wagon, scout ahead, or guard the rear? This arrangement is crucial for upcoming encounters and navigating the terrain.",
        'next_steps': {
            'driving the wagon': "Whenever the party is ready."
        }
    },
    {
        'name': "driving the wagon",
        'description': "The wagon, pulled by two oxen, contains various mining supplies and food worth 100 gp. The journey is uneventful initially, but the path is known for its dangers. The players must remain alert as they navigate the road.",
        'next_steps': {
            'finding horses': "At some point along the road, probably after some time has passed, the party encounters two dead horses blocking the path."
        }
    },
    {
        'name': "finding horses",
        'description': "As the party nears Phandalin, they encounter two dead horses blocking the path, riddled with black-feathered arrows.",
        'next_steps': {
            'combat with goblins': "Investigating the horses triggers the ambush from the goblins hiding in the thicket.",
        },
    },
    {
        'name': "combat with goblins",
        'description': "The party must quickly react to the goblin attack. Goblins, skilled in stealth and ambush tactics, launch their assault. The players must use their wits and combat skills to overcome this threat.",
        'next_steps': {
            'follow goblin trail': "If the party decides to track the goblins, they find a trail leading to the Cragmaw hideout."
        }
    },
    {
        'name': "follow goblin trail",
        'description': "The path is treacherous, with potential traps and signs of frequent goblin activity. Stealth and caution are advised.",
        'next_steps': {
            'cragmaw hideout': "They eventually alight on the hideout itself."
        }
    },
    {
        'name': "cragmaw hideout",
        'description': "The trail leads to a well-hidden cave, the Cragmaw hideout. The party must navigate through this dangerous lair, facing goblins and their traps. Their goal is to find Gundren, Sildar, or any information about their whereabouts.",
        'next_steps': {
            'rescue mission': "Attempt to rescue Gundren and Sildar, if they are found within the hideout.",
            'return to Phandalin': "After exploring the hideout, decide whether to return to Phandalin."
        }
    },
    {
        'name': "rescue mission",
        'description': "In the event Gundren or Sildar are found captive, the party must devise a plan to rescue them. This might involve combat, negotiation, or stealth. The fate of their mission and the lives of the captives hang in the balance.",
        'next_steps': {
            'return to Phandalin': "With or without the captives, make the journey back to Phandalin."
        }
    },
    {
        'name': "return to Phandalin",
        'description': "The journey concludes as the party arrives in Phandalin. They must report the outcome of their mission, deliver the supplies, and seek out Elmar Barthen at Barthen's Provisions. The town may offer more clues or quests related to the Cragmaw goblins and the fate of Gundren and Sildar.",
        'next_steps': {
            'explore Phandalin': "Explore the town of Phandalin for further adventures and quests."
        }
    }
]


story_part_name = 'start'

from modularity import OpenAI

client = OpenAI()

def summarize_plotline(messages):
    message_text = "\n".join([f"+ {x['role']}: {x['content']}" for x in messages])
    prompt = f"""
    Your goal is to summarize the plotpoints contained in the following conversation between a DM and a player.
    In each plot point, be as specific as possible.
    Keep note of any characters, locations, or items that are mentioned.
    Do not include any information not present in the following messages!

    Messages:
    {message_text}
    """
    print(prompt)

    messages = [
        {"role": "system", "content": prompt},
    ]

    response = get_response(messages)#, model="gpt-3.5-turbo")
    print('Summarized!')
    print(response)
    return response


M = []
summary = []

def respond():
    global M, summary, story_part_name, storyline

    story_part = [x for x in storyline if x['name'] == story_part_name][0]

    next_steps = "\n".join([f"\t{x}: {y}" for x, y in story_part['next_steps'].items()])

    my_messages = []

    prompt = f"""
You are a DM. 
You are currently in the scene "{story_part['name']}".
Although there is no rush to change the 'scene', you must eventually do so, in order to continue the story.

If you want to change the scene, type:
{{"type": "change_scene", "to": "scene name"}}

Otherwise, any (non-JSON) message you type will be sent to the player.

Description of the current scene:
    {story_part['description']}

Scenes available, their names and descriptions:
{next_steps}
    """
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
        print(response)

    # if so, change the scene
    else:
        story_part_name = response['to']
        print(f"Changed to scene: {story_part_name}")

        M.append({"role": "system", "content": f"Changed to scene: {story_part_name}"})

        # since the computer used its turn on changing the scene, let it go again (to respond with text)
        respond()

    # consolidate things, if it's getting long
    if len(M) > 10:
        # remember a summary of the messages
        summary.append(summarize_plotline(M))

        # clear the messages
        M = M[-2:]




# this is actually the interactive loop!
while True:
    # human does its thing
    query = input(">> ")
    M.append({"role": "user", "content": query})

    # computer does its thing
    respond()