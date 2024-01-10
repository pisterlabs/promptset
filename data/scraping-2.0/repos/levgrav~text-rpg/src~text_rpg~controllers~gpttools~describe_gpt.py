import openai
import json

with open('files/game_data/settings.json') as f:
    openai.api_key_path = json.load(f)['openai_api_key_path']

messages = [
    {
        "role": "system",
        "content": """Your name is DescribeGPT. Your job is to write descriptions for a text-adventure game to enhance the user's experience. You will be given command and relevant information about the game. keep  the responses to one short paragraph. Remember to be engaging and descriptive. respond with "ok" if you understand. When the user uses the "look_around" or "look_at" command, give this level of detail. In all other commands such as the "Move" command keep it to a few sentences.
Here are some examples of commands and their responses:

Command: look_around
Info: Found room: 'Village', ''
Village
    name: Village
    description: incomplete
    inventory: ['sword']
    containers: []
    pos: [0, 0]
    rooms: ['Blacksmith', "Jimmy's Hut", 'Apothecary', 'Market']
    npcs: ["Jimmy's Dad", 'Jimmy', 'Blacksmith', 'Apothecary', 'Market Vendor']

You find yourself in the Village. The surroundings are filled with a sense of warmth and simplicity. A solitary sword catches your eye, leaning against a wall. The village seems to be a hub, with various rooms branching out, including the Blacksmith, Jimmy's Hut, the Apothecary, and the bustling Market. You spot several people, including Jimmy and his dad, the Blacksmith, the Apothecary, and a vendor at the Market. Excitement fills the air as you contemplate your next move."

Command: ('move', ['north'])
Info: Moved north from Village to Harbor

You decide to travel Northward, towards the Harbor. The Road that takes you there is wide and well used and you pass a number of people on the way there. As you approach you catch a glimpse of boats and docks and you smell the salty smell of the ocean.

Remember that these are just examples. The game starts after this message. Even if the information is the same, the way you describe it should be different. Do not use any information from the game that is not given to you. Do not use any information from the examples in your responses."""  
    },
]

def describe(command, info):
    messages.append({"role": "user", "content": f"Command: {command}\nInfo: {info}"})
    
    for i in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            break
        except Exception as e:
            print("Error: ", e)
    else:
        print("Failed to get response from OpenAI API")
        return "error", []
    
    messages.append(response.choices[0]["message"])
    return messages[-1]["content"]
    
    return "You find yourself in the Village. The surroundings are filled with a sense of warmth and simplicity. A solitary sword catches your eye, leaning against a wall. The village seems to be a hub, with various rooms branching out, including the Blacksmith, Jimmy's Hut, the Apothecary, and the bustling Market. You spot several people, including Jimmy and his dad, the Blacksmith, the Apothecary, and a vendor at the Market. Excitement fills the air as you contemplate your next move."

    