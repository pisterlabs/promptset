import os
import time
import asyncio
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave.OpenAIClient import OpenAIClient
from alphawave.OSClient import OSClient
from alphawave_agents.Agent import Agent, AgentOptions
from NarrateCommand import NarrateCommand
from EndSceneCommand import EndSceneCommand
from CharacterCommand import CharacterCommand

# Create an OpenAI client
#client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)
#model = 'gpt-3.5-turbo'
client = OSClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)
model = 'chatglm2'

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
        "You can set the scene for a character but do not speak for the characters, let characters say their own lines.",
        "The dialog is being tracked behind the scenes so no need to pass it into the characters.",
        "\ncontext:",
        "\tperformance: {{$performance}}",
    ],
    prompt_options=PromptCompletionOptions(
        completion_type = 'chat',
        model = model,
        temperature = 0.01,
        max_input_tokens = 1200,
        max_tokens = 800,
    ),
    initial_thought={
        "reasoning": "This will make the experience more interactive and personalized, and also help me set the scene accordingly.",
        "plan": "- ask the user where to start the story from\n- use the narrate command to introduce the chosen scene. The introduction should be provided in a 'text' field in the JSON response\n- use the character commands to facilitate the dialog",
        "command": {
            "name": "ask",
            "input": {"question": initial_prompt}
        }
    },
    step_delay=5000,
    max_steps=50,
    logRepairs=True
)

# Create an agent
agent = Agent(options = agent_options)

# Add core commands to the agent
agent.addCommand(NarrateCommand())
agent.addCommand(EndSceneCommand())

# Define main characters
characters = ['Macbeth', 'Lady Macbeth','Banquo', 'King Duncan', 'Macduff', 'First Witch', 'Second Witch', 'Third Witch', 'Malcolm', 'Fleance', 'Hecate', 'Donalbain', 'Lady Macduff', 'Captain']
for name in characters:
    agent.addCommand(CharacterCommand(client, 'wizardLM', name))

# Define an additional 'extra' character to play minor roles
agent.addCommand(CharacterCommand(client, 'wizardLM', 'extra', 'use for minor characters or any missing commands'))

# Listen for new thoughts
#agent.events.on('newThought', lambda thought: print(f"[{thought['thoughts']['thought']}]"))

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
