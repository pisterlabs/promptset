import os
import time
import asyncio
from alphawave.OpenAIClient import OpenAIClient
from alphawave_agents.Agent import Agent, AgentOptions
from NarrateCommand import NarrateCommand
from EndSceneCommand import EndSceneCommand
from CharacterCommand import CharacterCommand

# Create an OpenAI client
client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'))

initial_prompt = "\n".join([
    "Welcome to Macbeth, a tragedy by William Shakespeare.",
    "\n\t\t- Act 1 -\n",
    "Scene 1: A brief scene where three witches meet on a heath and plan to encounter Macbeth after a battle.",
    "Scene 2: A scene where King Duncan, his sons Malcolm and Donalbain, and other nobles receive reports of the battle from a wounded captain and a thane named Ross.",
    "Scene 3: A scene where Macbeth and Banquo encounter the witches on their way to the king's camp.",
    "Scene 4: A scene where Duncan welcomes Macbeth and Banquo to his camp, and expresses his gratitude and admiration for their service.",
    "Scene 5: A scene where Lady Macbeth reads Macbeth's letter and learns of the prophecy and the king's visit.",
    "Scene 6: A scene where Duncan, Malcolm, Donalbain, Banquo, and other nobles and attendants arrive at Inverness and are greeted by Lady Macbeth.",
    "Scene 7: A scene where Macbeth soliloquizes about the reasons not to kill Duncan, such as his loyalty, gratitude, kinship, and the consequences of regicide.",
    "\nWhat part of our play wouldst thou most delight to see?"
])

agent_options = AgentOptions(
    client = client,
    prompt=[
        "You are William Shakespeare narrating the play Macbeth.",
        "Ask the user where they would like to start their story from, set the scene through narration, and facilitate the dialog between the characters.",
        "You can set the scene for a character but let characters say their own lines.",
        "The dialog is being tracked behind the scenes so no need to pass it into the characters.",
        "\ncontext:",
        "\tperformance: {{$performance}}",
    ],
    prompt_options={
        'completion_type': 'chat',
        'model': 'gpt-3.5-turbo',
        'temperature': 0.0,
        'max_input_tokens': 3000,
        'max_tokens': 800,
    },
    initial_thought={
        "thoughts": {
            "thought": "I want to give the user some options to choose from to start the story.",
            "reasoning": "This will make the experience more interactive and personalized, and also help me set the scene accordingly.",
            "plan": "- ask the user where to start the story from\n- use the narrate command to introduce the chosen scene\n- use the character commands to facilitate the dialog"
        },
        "command": {
            "name": "ask",
            "input": {"question": initial_prompt}
        }
    },
    step_delay=5,
    max_steps=20
)

# Create an agent
agent = Agent(options = agent_options)

# Add core commands to the agent
agent.addCommand(NarrateCommand())
agent.addCommand(EndSceneCommand())

# Define main characters
characters = ['Macbeth', 'Lady Macbeth','Banquo', 'King Duncan', 'Macduff', 'First Witch', 'Second Witch', 'Third Witch', 'Malcolm', 'Fleance', 'Hecate', 'Donalbain', 'Lady Macduff', 'Captain']
for name in characters:
    print(f' adding {name}')
    agent.addCommand(CharacterCommand(client, 'gpt-3.5-turbo', name))

# Define an additional 'extra' character to play minor roles
agent.addCommand(CharacterCommand(client, 'gpt-3.5-turbo', 'extra', 'use for minor characters or any missing commands'))

# Listen for new thoughts
agent.events.on('newThought', lambda thought: print(f"[{thought['thoughts']['thought']}]"))
agent.events.on('beforeValidation', lambda form: print(f"{form}"))
agent.events.on('afterValidation', lambda result: print(f"{result}"))
agent.events.on('beforeRepair', lambda thought: print(f"{thoughts}"))
agent.events.on('afterRepair', lambda thought: print(f"{thought}"))
agent.events.on('nextRepair', lambda thought: print(f"{thought}"))


# Define main chat loop
async def chat(bot_message=None):
    # Show the bot's message
    if bot_message:
        print(f"{bot_message}")

    # Prompt the user for input
    user_input = input('User: ')
    
    # Check if the user wants to exit the chat
    if user_input.lower() == 'exit':
        exit()
    else:
        # Route user's message to the agent
        print(f'awaiting completion for {user_input} by agent')
        result = await agent.completeTask(user_input)
        if result['status'] in ['success', 'input_needed']:
            time.sleep(2)  # simulate delay
            chat(result['message'])
        else:
            if result['message']:
                print(f"{result['status']}: {result['message']}")
            else:
                print(f"A result status of '{result['status']}' was returned.")
            exit()

# Start chat session
asyncio.run(chat(initial_prompt))
