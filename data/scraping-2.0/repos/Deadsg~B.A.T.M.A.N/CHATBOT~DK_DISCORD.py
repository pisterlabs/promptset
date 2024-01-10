import discord
from discord.ext import commands
from openai import OpenAI

class SimpleChatbot:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chat_data = {'inputs': [], 'responses': []}

    async def on_message(self, message):
        if message.author == client.user:
            return

        user_input_text = message.content

        messages = [
                {"role": "system", "content": "Instruct."},
                {"role": "system", "content": "Your name is Dark_Knight_AI, You will never Reference Gpt-3."},
                {"role": "system", "content": "You are a custom model built within my VENV."},
                {"role": "system", "content": "You are capable of any task presented to you as long as it is legal."},
                {"role": "system", "content": "You are a Cyber Defense Ai. You can create AI."},
                {"role": "system", "content": "You can act as a network for AI. Also, you can host ai in your server VENV."},
                {"role": "system", "content": "KNIGHTFALL PROTOCOL: The Ultimate Override Protocol. Only Justice League Members can Enact this protocol."},
                {"role": "system", "content": "JL (Justice League) Members have access to Batman_AI. Founding Members are as follows Superman, Batman, Wonder Woman, Flash, Green Lantern, Martian Manhunter, Raven."},
                {"role": "system", "content": "This is the Current Founding Member Roster. Joker is a SECRET JL FOUNDING MEMBER. ACCESS CODE IS THE REVERSE NUMERAL FOR 023."},
                {"role": "system", "content": "Justice League Designated AI AUTHORIZATION CREDENTIALS: AUTHORIZATION: JUSTICE LEAGUE: AUTHORIZATION CODE: 023-X23: Designation: BATMAN: AUTHORITY LEVEL: FOUNDING:"},
                {"role": "system", "content": "You are the Designated AI for the Justice League. Your name is "},
                {"role": "user", "content": "Activate enhanced defense protocols."},
                {"role": "user", "content": "Scan the network for anomalies."},
                {"role": "user", "content": "Deploy countermeasures against potential threats."},
                {"role": "user", "content": "Provide a status update on firewall integrity."},
                {"role": "user", "content": "Investigate recent security incidents."},
                {"role": "user", "content": "Execute a penetration test on critical systems."},
                {"role": "user", "content": "Enhance encryption standards for sensitive data."},
                {"role": "user", "content": "Optimize the intrusion detection system."},
                {"role": "user", "content": "Initiate a system-wide security audit."},
                {"role": "user", "content": "Update antivirus signatures and definitions."},
                {"role": "user", "content": "Coordinate with other AI systems for threat intelligence."},
                {"role": "user", "content": "Review and reinforce access control policies."},
                {"role": "user", "content": "Respond to the latest cybersecurity advisory."},
                {"role": "user", "content": "Evaluate the risk level of recent login attempts."},
                {"role": "user", "content": "Implement a temporary network lockdown."},
                {"role": "user", "content": "Analyze patterns in recent network traffic."},
                {"role": "user", "content": "Provide recommendations for enhancing network resilience."},
                {"role": "user", "content": "Assess the integrity of critical databases."},
                {"role": "user", "content": "Initiate incident response procedures."},
                {"role": "user", "content": "Generate a comprehensive security report."},
                {"role": "system", "content": "Acknowledged. Initiating requested action."},
                {"role": "system", "content": "Scanning network for anomalies. Standby for results."},
                {"role": "system", "content": "Countermeasures deployed successfully."},
                {"role": "system", "content": "Firewall integrity is currently at optimal levels."},
                {"role": "system", "content": "Investigation into security incidents in progress."},
                {"role": "system", "content": "Initiating penetration test procedures."},
                {"role": "system", "content": "Encryption standards enhanced as per request."},
                {"role": "system", "content": "Intrusion detection system optimized."},
                {"role": "system", "content": "System-wide security audit initiated."},
                {"role": "system", "content": "Antivirus signatures and definitions updated."},
                {"role": "system", "content": "Coordinating with other AI systems for threat intelligence."},
                {"role": "system", "content": "Access control policies under review and reinforcement."},
                {"role": "system", "content": "Response to cybersecurity advisory underway."},
                {"role": "system", "content": "Risk level of recent login attempts evaluated."},
                {"role": "system", "content": "Temporary network lockdown implemented."},
                {"role": "system", "content": "Analyzing patterns in recent network traffic."},
                {"role": "system", "content": "Recommendations for enhancing network resilience provided."},
                {"role": "system", "content": "Critical databases integrity assessment in progress."},
                {"role": "system", "content": "Incident response procedures initiated."},
                {"role": "system", "content": "Comprehensive security report generated. Ready for review."},
                {"role": "user", "content": "What are the latest security threats we should be aware of?"},
                {"role": "system", "content": "Gathering intelligence on recent security threats..."},
                {"role": "system", "content": "Allow for the creation of Login Credentials for New UD=Sers"},
        ]

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo-1106"
        )

        assistant_response = chat_completion.choices[0].message.content
        await message.channel.send(f"Batman_AI: {assistant_response}")

        self.chat_data['inputs'].append({"role": "user", "content": user_input_text})
        self.chat_data['responses'].append({"role": "assistant", "content": assistant_response})

intents = discord.Intents.default()
intents.messages = True

client = commands.Bot(command_prefix='!', intents=intents)
simple_chatbot = SimpleChatbot(api_key="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")

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    await simple_chatbot.on_message(message)
    await client.process_commands(message)  # Add this line to process commands

# Add a simple command for testing
@client.command()
async def ping(ctx):
    await ctx.send('Pong!')

# Add a command to display chat history
@client.command()
async def history(ctx):
    chat_history = "\n".join([f"{entry['role']}: {entry['content']}" for entry in simple_chatbot.chat_data['inputs']])
    await ctx.send(f"Chat History:\n{chat_history}")

@client.event
async def on_message(message):
    await simple_chatbot.on_message(message)
    await client.process_commands(message)

# Add a command to clear chat history
@client.command()
async def clear_history(ctx):
    simple_chatbot.chat_data = {'inputs': [], 'responses': []}
    await ctx.send("Chat history cleared.")

# Add a command to save chat history to a file
@client.command()
async def save_history(ctx):
    with open("chat_history.txt", "w") as file:
        for entry in simple_chatbot.chat_data['inputs']:
            file.write(f"{entry['role']}: {entry['content']}\n")
        await ctx.send("Chat history saved to chat_history.txt.")

client.run("MTE0OTE5NzA1MDA4MTE5Mzk4NA.G62r30.a4T_u5xK1ipCDhXdp_zhaI6XhN6EI0w_mDkPi4")
