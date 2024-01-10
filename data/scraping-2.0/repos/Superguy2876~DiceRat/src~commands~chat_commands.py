import random
import os
import uuid
from urllib import response
import lightbulb
import hikari
from lightbulb import commands
import redis
from openai import OpenAI
from dotenv import load_dotenv, dotenv_values
import json
load_dotenv()




def send_and_receive(client, messages):
    return client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages
    )

# get adjudicators from redis
# key format: gods:<name>
def get_adjudicators():
    r =  redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
    adjudicators = [name.split(':')[1] for name in r.scan_iter("gods:*")]
    # close connection
    r.close()
    return adjudicators

def get_adjudicators(redis_pool):
    r = redis.Redis(connection_pool=redis_pool)
    # get name keys then get description values, return both as a list of tuples
    adjudicators = [(name.split(':')[1], r.get(name)) for name in r.scan_iter("god:*")]
    # close connection
    r.close()
    return adjudicators

    
    

@lightbulb.option("password", "The password to use.", str, required=True)
@lightbulb.command("toggle_spr", "Activate the Custom Scissors, Paper, Rock Game.")
@lightbulb.implements(commands.SlashCommand)
async def toggle_spr(ctx: lightbulb.context.Context) -> None:
    password = ctx.options.password

    config = dotenv_values('.env')
    if password != config['SPR_ACTIVATE_KEY']:
        await ctx.respond("Incorrect password.", flags=hikari.MessageFlag.EPHEMERAL)
        return
    
    guild_id = ctx.guild_id
    key = f"{guild_id}:SPR_active"

    r = redis.Redis(connection_pool=ctx.bot.redis_pool)
    
    if r.get(key) is None or r.get(key) == "False":
        r.set(key, "True")
        await ctx.respond("SPR activated!", flags=hikari.MessageFlag.EPHEMERAL)
        return

    r.set(key, "False")

    await ctx.respond("SPR deactivated!", flags=hikari.MessageFlag.EPHEMERAL)

@lightbulb.option("choice", "Choose your object.", str, required=True)
@lightbulb.option("adjudicator", "Choose your adjudicator.", int, required=True)
@lightbulb.command("spr", "Enter a Custom Scissors, Paper, Rock Game.")
@lightbulb.implements(commands.SlashCommand)
async def scissors_paper_rock(ctx: lightbulb.context.Context) -> None:
    r = redis.Redis(connection_pool=ctx.bot.redis_pool)
    # Check if the game is active.
    guild_id = ctx.guild_id
    key = f"{guild_id}:SPR_active"

    if r.get(key) is None or r.get(key) == "False":
        await ctx.respond("SPR is not active.", flags=hikari.MessageFlag.EPHEMERAL)
        return

    user_choice = ctx.options.choice.lower()
    adjudicator_index = int(ctx.options.adjudicator) - 1

    # Load the list of adjudicators from Redis
    adjudicators = get_adjudicators(ctx.bot.redis_pool)

    if adjudicator_index < 0 or adjudicator_index >= len(adjudicators):
        await ctx.respond(f"No adjudicator found with the number {adjudicator_index + 1}.", flags=hikari.MessageFlag.EPHEMERAL)
        return

    adjudicator_name = adjudicators[adjudicator_index][0]

    guild_id = ctx.guild_id
    key = f"{guild_id}:SPR:{adjudicator_name}"

    r = redis.Redis(connection_pool=ctx.bot.redis_pool)

    existing_challenge = r.get(key)

    if existing_challenge is None:
        r.set(key, user_choice)
        await ctx.respond("SPR challenge started! Waiting for another player.")
        return
    
    # You may want to clear the key after the game is done.
    r.delete(key)

    # If we're here, it means that there is an existing challenge.
    await ctx.respond("A challenge has begun. Please wait for the results.")
    players = [existing_challenge, user_choice]

    # We can use the OpenAI API to generate a response.
    # You can use the `send_and_receive` function above to do this.
    # Setup the system first.
    client = OpenAI()

    system_message = {'role':'system'}
    formatting = """\nUse the following format:
    Player 1: <object>
    Player 2: <object>
    Winner: <player>
    Reason: <reason>"""
    # get adjudicator description from redis
    system_message['content'] = r.get(f"god:{adjudicator_name}") + formatting
    user_message = {'role':'user'}

    user_message['content'] = f"player 1's choice is {players[0]}\nplayer 2's choice is {players[1]}"
    messages = [system_message, user_message] 
    completion = send_and_receive(client, messages)
    response = completion.choices[0].message.content

    if len(response) > 2000:
        # create txt file and put response in it
        filename = f"{uuid.uuid4()}_response.txt"
        with open(filename, 'w') as wf:
            wf.write(response)
        
        # create new response, attach file to it, and send
        response = f"Response body too long. See text file for response."
        file = hikari.File(filename)
        await ctx.edit_last_response(response, attachment=file)
        os.remove(filename)
        return

    # Edit the message with the response.
    await ctx.edit_last_response(response)
    
@lightbulb.command("adjudicators", "List possible Adjudicators of SPR.", ephemeral=True)
@lightbulb.implements(commands.SlashCommand)
async def adjudicators(ctx: lightbulb.context.Context) -> None:
    r = redis.Redis(connection_pool=ctx.bot.redis_pool)
    
    adjudicators = get_adjudicators(ctx.bot.redis_pool)

    await ctx.respond('\n\n'.join([f"{i+1}: {adjudicator[0]}: {adjudicator[1]}" for i, adjudicator in enumerate(adjudicators)]))

def load(bot: lightbulb.BotApp) -> None:
    bot.command(toggle_spr)
    bot.command(scissors_paper_rock)
    bot.command(adjudicators)

    # load gods into redis
    r = redis.Redis(connection_pool=bot.redis_pool)
    # print working directory
    print(os.getcwd())
    with open('data/json/gods.json') as f:
        gods = json.load(f)
        for god in gods:
            # key format: god:<name>
            r.set(f"god:{god['name']}", god['description'])

def unload(bot: lightbulb.BotApp) -> None:
    bot.remove_command(bot.get_slash_command("toggle_spr"))
    bot.remove_command(bot.get_slash_command("spr"))
    bot.remove_command(bot.get_slash_command("adjudicators"))