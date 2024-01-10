import datetime
import os
from dotenv import load_dotenv
import discord
import pandas as pd
from discord.ext import commands
import openai
import csv
import asyncio
import re
from collections import defaultdict
from pathlib import Path
from transformers import AlbertForQuestionAnswering, AlbertTokenizer
import torch
import json
import aiohttp


# Read the CSV file
df = pd.read_csv('chat_history.csv')

# Convert DataFrame to JSON
json_data = df.to_json(orient='records')

# Save JSON data to a file
with open('output.json', 'w') as file:
    file.write(json_data)



# Load the environment variables from the .env file
load_dotenv()
print("Loaded environment variables.")

# Getting the Discord Bot Token from environment variables
TOKEN = os.environ.get('DISCORD_BOT_TOKEN')

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

model_name = 'albert-base-v2'  # Replace with the desired ALBERT model
model = AlbertForQuestionAnswering.from_pretrained(model_name)
tokenizer = AlbertTokenizer.from_pretrained(model_name)

def create_empty_chat_history():
    return pd.DataFrame(columns=['timestamp', 'author', 'content'])

#chat_histories = defaultdict(create_empty_chat_history)
chat_histories = {}

# Loading the chat history from the CSV file into the DataFrame
chat_history = pd.read_csv('chat_history.csv', names=[
                           'timestamp', 'author', 'content'], skiprows=1)
chat_history['timestamp'] = pd.to_datetime(
    chat_history['timestamp'], format="%Y-%m-%d %H:%M:%S.%f%z")
chat_history = chat_history.dropna()  # remove blank lines
print("Loaded chat history from CSV.")

recall_role_instructions = "You are a helpful assistant that finds information in a conversation and answers user's questions about what has occurred or been said in this chat"
summarize_role_instructions = "You are a helpful assistant that summarizes a conversation."

def load_chat_history(channel_id):
    csv_file = f'chat_history_{channel_id}.csv'
    if Path(csv_file).exists():  # check if the file exists
        # Load the chat history from the CSV file into the DataFrame
        chat_history = pd.read_csv(csv_file, names=['timestamp', 'author', 'content'], skiprows=1, encoding='ISO-8859-1')
        chat_history['timestamp'] = pd.to_datetime(chat_history['timestamp'], format="%Y-%m-%d %H:%M:%S.%f%z")
        chat_history = chat_history.dropna()  # remove blank lines
        print(f"Loaded chat history for channel {channel_id} from CSV.")
    else:
        chat_history = create_empty_chat_history()
        print(f"No existing chat history for channel {channel_id}. Created new chat history.")
    return chat_history

async def answer_question(query, text):
    inputs = tokenizer.encode_plus(query, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.decode(input_ids[answer_start:answer_end], skip_special_tokens=True)

    return answer

async def generate_summary(text, query=None):
    if query:
        user_text = f"Please provide a summary of the following conversation, focusing on {query}:\n\n{text}"
    else:
        user_text = f"Please provide a summary of the following conversation:\n\n{text}"

    inputs = tokenizer.encode_plus(user_text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary

async def answer_question(query, text):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    def sync_request():
        print("Generating OpenAI Request...")
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": recall_role_instructions},
                {"role": "user",
                    "content": f"Please answer the question or tell me what we discussed regarding ( {query} ) within this conversation:\n\n{text}"}
            ],
            max_tokens=150,
            n=1,
            temperature=0.7,
        )

    loop = asyncio.get_event_loop()
    print("Executing OpenAI Request...")
    response = await loop.run_in_executor(None, sync_request)
    answer = response.choices[0].message['content'].strip()
    print(f"Answer: {answer}")
    return answer

async def generate_summary(text, query=None):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    if query:
        user_text = f"Please provide a summary of the following conversation, focusing on {query}:\n\n{text}"
    else:
        user_text = f"Please provide a summary of the following conversation:\n\n{text}"

    def sync_request():
        print("Generating OpenAI Request for Summary...")
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": summarize_role_instructions},
                {"role": "user", "content": user_text}
            ],
            max_tokens=150,
            n=1,
            temperature=0.7,
        )

    loop = asyncio.get_event_loop()
    print("Executing OpenAI Request for Summary...")
    response = await loop.run_in_executor(None, sync_request)
    summary = response.choices[0].message['content'].strip()
    print(f"Summary: {summary}")
    return summary

from flask import Flask
app = Flask(__name__)

@app.route('/')
def dockerconfirm():
    return 'Woohoo! Docker container is successfully running on this instance.'

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f"Current Chat History: {chat_history}")

    if not os.path.isfile('chat_history.csv'):
        with open('chat_history.csv', 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['timestamp', 'author', 'content'])

    for guild in bot.guilds:  # iterate over all servers the bot is connected to
        for channel in guild.channels:  # iterate over all channels in the server
            if isinstance(channel, discord.TextChannel):  # make sure it's a text channel
                # Load the chat history for the channel
                chat_histories[channel.id] = load_chat_history(channel.id)
    print("Loaded all chat histories.")
    
@bot.event
async def on_message(message):
    global chat_histories

    if message.author == bot.user:
        return

    content = message.content

    # Get the chat history for the current channel
    chat_history = chat_histories[message.channel.id]

    # If the message is not a command
    if not content.startswith('!'):
        # Append the new message to the chat history DataFrame with the timestamp
        chat_history.loc[len(chat_history)] = {
            'timestamp': message.created_at, 'author': message.author.name, 'content': content}

        print(f"Added message to chat history of channel {message.channel.id}")

        # Append the new message to the CSV file
        with open(f'chat_history_{message.channel.id}.csv', 'a', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                [message.created_at, message.author.name, content])

        print(f"Added message to CSV for channel {message.channel.id}")

        # Update the chat_histories dictionary
        chat_histories[message.channel.id] = chat_history

    await bot.process_commands(message)

@bot.command(name='recall')
async def recall(ctx, *args):
    query = ' '.join(args)
    max_tokens = 3000

    chat_history = chat_histories[ctx.channel.id]

    relevant_history = chat_history[
        ~(chat_history['author'] == bot.user.name) & 
        ~(chat_history['content'].str.startswith('!')) & 
        (chat_history['content'].str.contains('|'.join(args), case=False, na=False))
    ].tail(10)
    print(f"relevant_history message{relevant_history}")

    # Estimate token count
    estimated_tokens = relevant_history['content'].apply(lambda x: len(x.split()))

    # Split messages into two parts if estimated token count exceeds the limit
    if estimated_tokens.sum() > max_tokens:
        # Find the index where the cumulative sum of tokens exceeds the limit
        split_index = estimated_tokens.cumsum().searchsorted(max_tokens)[0]

        # Split the messages
        part1 = "\n".join(
            f"{row.timestamp} - {row.author}: {row.content}" for _, row in relevant_history.iloc[:split_index].iterrows()
        )
        part2 = "\n".join(
            f"{row.timestamp} - {row.author}: {row.content}" for _, row in relevant_history.iloc[split_index:].iterrows()
        )

        # Send two recall requests
        summary1 = await answer_question(query, part1)
        summary2 = await answer_question(query, part2)
        summary = f"Part 1: {summary1}\nPart 2: {summary2}"
    else:
        conversation_text = "\n".join(
            f"{row.timestamp} - {row.author}: {row.content}" for _, row in relevant_history.iterrows()
        )

        summary = await answer_question(query, conversation_text)

    await ctx.send(summary)


@bot.command(name='summarize')
async def summarize(ctx, *args):
    query = ' '.join(args) if args else None

    chat_history = chat_histories[ctx.channel.id]

    relevant_history = chat_history[~(chat_history['author'] == bot.user.name) & ~(
        chat_history['content'].str.startswith('!'))]

    conversation_text = "\n".join(
        f"{row.timestamp} - {row.author}: {row.content}" for _, row in relevant_history.iterrows())

    summary = await generate_summary(conversation_text, query)
    await ctx.send(summary)
   
    
# Testing
@bot.command(name='test')
async def test_command(ctx, *args):
    print("Test command invoked.")
    
    # Process the test command arguments
    if len(args) == 0:
        await ctx.send("Please provide test arguments.")
        return

    # Perform the desired test actions
    # ...

    # Send the test results or output
    # ...
    
    print("Test command executed.")



# Add Task
task_list = []  # Declare an empty task list

@bot.command(name='addtask')
async def add_task(ctx):
    if any(task['description'] == 'Please provide a task description.' for task in task_list):
        # A task creation is already in progress for this user and channel
        await ctx.send("A task creation is already in progress.")
        return

    # Ask for the task description
    await ctx.send("Please provide a task description.")

    def check(message):
        return message.author == ctx.author and message.channel == ctx.channel

    try:
        # Wait for the user's response
        description_message = await bot.wait_for('message', check=check, timeout=30)

        # Get the task description from the user's response
        task_description = description_message.content

        # Ask for the assignee
        await ctx.send("Please provide the assignee for the task.")

        # Wait for the user's response
        assignee_message = await bot.wait_for('message', check=check, timeout=30)

        # Get the assignee from the user's response
        assignee = assignee_message.content

        # Ask for the deadline
        await ctx.send("Please provide the deadline for the task.")

        # Wait for the user's response
        deadline_message = await bot.wait_for('message', check=check, timeout=30)

        # Get the deadline from the user's response
        deadline = deadline_message.content

        # Create a new task with the provided details and the user who created it
        task = {
            'description': task_description,
            'assignee': assignee,
            'deadline': deadline,
            'status': 'In Progress',
            'created_by': ctx.author.name  # Add the created_by field
        }

        # Add the task to the task list
        task_list.append(task)

        await ctx.send("Task added successfully.")

    except asyncio.TimeoutError:
        await ctx.send("You took too long to respond. Task creation canceled.")


from discord.ext import commands

task_list = []  # Initialize an empty task list

# Your existing view_task function
@bot.command(name='viewtask')
async def view_task(ctx):
    if not task_list:
        await ctx.send("The task list is empty.")
    else:
        for index, task in enumerate(task_list, start=1):
            # Display the task information
            task_info = f"Task {index}:\n" \
                        f"Description: {task['description']}\n" \
                        f"Assignee: {task['assignee']}\n" \
                        f"Deadline: {task['deadline']}\n" \
                        f"Status: {task['status']}\n"

            await ctx.send(task_info)

@bot.command(name='updatetask')
async def update_task(ctx, task_index: int, **updates):
    global task_list  # Access the global task_list variable

    if not task_list or task_index <= 0 or task_index > len(task_list):
        await ctx.send("Invalid task index.")
        return

    task = task_list[task_index - 1]
    if not updates:
        await ctx.send("Please provide at least one update.")
        return

    # Update the task information with the provided updates
    for key, value in updates.items():
        if key in task:
            task[key] = value
        else:
            await ctx.send(f"Invalid key '{key}'. Skipping update for this key.")

    await ctx.send("Task updated successfully.")





    
#@bot.command(name='removetask')
#async def remove_task(ctx, project_name, task_id):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task ID is provided
    #if not task_id:
        #await ctx.send("Please provide a task ID.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Remove the task from the project
    # ...

    #await ctx.send(f"Task {task_id} removed successfully from project '{project_name}'.")
    
#@bot.command(name='completetask')
#async def complete_task(ctx, project_name, task_id):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task ID is provided
    #if not task_id:
        #await ctx.send("Please provide a task ID.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Mark the task as completed
    # ...

    #await ctx.send(f"Task {task_id} marked as completed in project '{project_name}'.")
    
#@bot.command(name='assigntask')
#async def assign_task(ctx, project_name, task_id, new_assignee):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task ID is provided
    #if not task_id:
        #await ctx.send("Please provide a task ID.")
        #return

    # Check if the new assignee is provided
    #if not new_assignee:
        #await ctx.send("Please provide a new assignee for the task.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Change the assignee of the task
    # ...

    #await ctx.send(f"Assignee of task {task_id} updated in project '{project_name}'.")


#@bot.command(name='changetaskdeadline')
#async def change_task_deadline(ctx, project_name, task_id, new_deadline):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task ID is provided
    #if not task_id:
        #await ctx.send("Please provide a task ID.")
        #return

    # Check if the new deadline is provided
    #if not new_deadline:
        #await ctx.send("Please provide a new deadline for the task.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Change the deadline of the task
    # ...

    #await ctx.send(f"Deadline of task {task_id} updated in project '{project_name}'.")
    
# @bot.command(name='viewarchives')
# async def view_archives(ctx):
#     # Retrieve a list of archived projects from your data storage
#     archived_projects = retrieve_archived_projects()

#     if not archived_projects:
#         await ctx.send("There are no archived projects.")
#         return

#     # Generate a formatted message with the archived projects
#     message = "Archived Projects:\n"
#     for project_name in archived_projects:
#         message += f"- {project_name}\n"

#     # Send the formatted message to the channel
#     await ctx.send(message)


#@bot.command(name='restoreproject')
#async def restore_project(ctx, project_name):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the project exists in the archived projects
    #if project_name not in archived_projects:
        #await ctx.send("Project not found in archived projects.")
        #return

    # Restore the project by moving it from archived to active projects
    #project = archived_projects[project_name]
    #active_projects[project_name] = project
    #del archived_projects[project_name]

    # Send a confirmation message
    #await ctx.send(f"Project '{project_name}' has been restored.")
    
    
#@bot.command(name='setpriority')
#async def set_priority(ctx, project_name, task_id, priority_level):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task ID is provided
    #if not task_id:
        #await ctx.send("Please provide a task ID.")
        #return

    # Check if the priority level is provided
    #if not priority_level:
        #await ctx.send("Please provide a priority level.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Get the project from the projects list or database
    #project = projects[project_name]

    # Find the task in the project's task list based on the task ID
    #task = None
    #for t in project['tasks']:
        #if t['task_id'] == task_id:
            #task = t
            #break

    # Check if the task exists
    #if not task:
        #await ctx.send("Task not found.")
        #return

    # Set the priority level of the task
    #task['priority'] = priority_level

    #await ctx.send(f"Priority level of Task {task_id} in Project {project_name} set to {priority_level}.")

#@bot.command(name='viewteam')
#async def view_team(ctx):
    # Implementation to display information about the project management team members
    # ...

    # Retrieve the team members from your data storage
    # team_members = retrieve_team_members()

    # Generate a formatted message with the team members' information
    # message = "Project Management Team Members:\n"
    # for member in team_members:
    #     member_name = member['name']
    #     member_role = member['role']
    #     message += f"- {member_name} ({member_role})\n"

    # Send the formatted message to the channel
    # await ctx.send(message)
    
    
#@bot.command(name='assignrole')
#async def assign_role(ctx, user, role):
    # Check if the user is a member of the project management team
    #if not is_team_member(user):
        #await ctx.send(f"{user} is not a member of the project management team.")
        #return

    # Assign the specified role to the user
    #if assign_role_to_user(user, role):
        #await ctx.send(f"Role '{role}' assigned to {user} successfully.")
    #else:
        #await ctx.send("Failed to assign role. Please check the provided role and try again.")


bot.run(TOKEN)
