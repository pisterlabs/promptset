import discord
import asyncio
import subprocess
import keys
import re
import json
from openai import OpenAI
from discord import app_commands

# Initialize Discord Client and OpenAI Client
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
openai_client = OpenAI(api_key=keys.openai_api_key)

# File Definitions
instructions_file = 'instructions.json'
chat_history_file = 'chat_history.json'
alerts_file = 'alerts.json'
threat_intel_file = 'threat_intel.json'
def run_hacker_news_script():
    print("Running 'hacker_news_script.py' to summarize Hacker News RSS feed.")
    subprocess.run(["python", "hacker_news_script.py"])
def run_script(script_name):
    print(f"Running '{script_name}'")
    subprocess.run(["python", script_name])

def estimate_tokens(text):
    return len(text.split())

def estimate_word_count(chat_history):
    return sum(len(message["content"].split()) for message in chat_history)

def read_json_from_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []
def write_json_to_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def run_summarize_alerts_script():
    print("Running 'summarizealerts.py' to summarize alerts.")
    subprocess.run(["python", "summarizealerts.py"])
def run_forget_script():
    print("Running 'forget some stuff.py' to manage chat history size.")
    subprocess.run(["python", "forget_some_stuff.py"])
def estimate_word_count(chat_history):
    return sum(len(message["content"].split()) for message in chat_history)
async def call_level12_summary_script(interval=70):
    await client.wait_until_ready()
    while not client.is_closed():
        subprocess.run(["python", "level12andupsummary.py"])
        await asyncio.sleep(interval)
def run_vuln_script(agent_id):
    print(f"Running vulnerability check for agent {agent_id}.")
    subprocess.run(["python", "agentvuln3.py", agent_id])
async def call_event_summary_script(interval=60):
    await client.wait_until_ready()
    while not client.is_closed():
        subprocess.run(["python", "event651summary.py"])
        await asyncio.sleep(interval)
async def call_hacker_news_script(interval=14400):
    await client.wait_until_ready()
    while not client.is_closed():
        run_hacker_news_script()
        await asyncio.sleep(interval)
def read_alerts_from_file(filename=alerts_file):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []
def write_alerts_to_file(alerts, filename=alerts_file):
    with open(filename, 'w') as file:
        json.dump(alerts, file, indent=4)

def read_instructions(filename=instructions_file):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []
def read_chat_history_from_file(filename=chat_history_file):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []
def write_chat_history_to_file(chathistory, filename=chat_history_file):
    with open(filename, 'w') as file:
        json.dump(chathistory, file, indent=4)

chathistory = read_chat_history_from_file()

def chat_completion(message):
    instructions = read_json_from_file(instructions_file)
    threat_news = read_json_from_file(threat_intel_file)
    alerts = read_json_from_file(alerts_file)
    chat_history = read_chat_history_from_file()

    cleaned_message_content = re.sub(r"<.*?>", "", message.content)
    user_message = {"role": "user", "content": cleaned_message_content}
    chat_history.append(user_message)
    combined_history = instructions + threat_news + alerts + chat_history
    combined_text = " ".join([item["content"] for item in combined_history])
    estimated_tokens = estimate_tokens(combined_text)
    print(f"Estimated number of tokens for this prompt: {estimated_tokens}")
    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=combined_history,
        temperature=0.5,
    )
    reply = response.choices[0].message.content
    bot_response = {"role": "assistant", "content": reply}
    chat_history.append(bot_response)
    write_chat_history_to_file(chat_history)
    word_count = estimate_word_count(chat_history)
    if word_count > 10000:
        run_forget_script()
    return reply

async def send_message_chunks(channel, message):
    while message:
        chunk, message = message[:2000], message[2000:]
        await channel.send(chunk)

# Discord Commands
@tree.command(name="getvulns", description="Get vulnerabilities for a specific agent")
@app_commands.choices(agent_id=[
    app_commands.Choice(name="Agent 000", value="000"),
    app_commands.Choice(name="Agent 001", value="001"),
    app_commands.Choice(name="Agent 002", value="002"),
    app_commands.Choice(name="Agent 003", value="003"),
    app_commands.Choice(name="Agent 004", value="004"),
    app_commands.Choice(name="Agent 005", value="005"),
    app_commands.Choice(name="Agent 006", value="006"),
    # Add more choices as needed
])
async def get_vulns(interaction: discord.Interaction, agent_id: str):
    await interaction.response.defer()
    run_vuln_script(agent_id)
    await interaction.followup.send(f"Vulnerabilities checked for agent {agent_id}.")

@tree.command(name="sumalerts", description="Summarize the alerts in alerts.json")
async def sumalerts_command(interaction):
    await interaction.response.defer()
    run_summarize_alerts_script()
    await interaction.followup.send("Alerts summarized.")
@tree.command(name="forget", description="Run forget_some_stuff script")
async def first_command(interaction):
    await interaction.response.defer()
    run_forget_script()
    await interaction.followup.send("Forget script executed.")

# Discord Events
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if client.user.mentioned_in(message) or 'Heimdall' in message.content:
        async with message.channel.typing():
            reply = chat_completion(message)
        if len(reply) > 2000:
            await send_message_chunks(message.channel, reply)
        else:
            await message.channel.send(reply)
@client.event
async def on_ready():
    await tree.sync()
    print("Bot is ready and commands are synced globally.")
    client.loop.create_task(call_event_summary_script())
    client.loop.create_task(call_level12_summary_script())
    client.loop.create_task(call_hacker_news_script())

client.run(keys.discord_token)
