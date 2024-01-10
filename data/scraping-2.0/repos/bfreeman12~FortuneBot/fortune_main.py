import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from os import environ
from random import Random,randrange
import random
from PIL import Image, ImageDraw, ImageFont
from math import ceil
from subprocess import getoutput
import aiohttp
import openai
from openai import OpenAI
#gathers token for running the bot
load_dotenv('token.env')
token = environ["DISCORD_TOKEN"]
GPT_API_KEY = environ['OPENAI_API_KEY']
client = OpenAI(api_key=environ['OPENAI_API_KEY'])

#defines prefix and intents
bot = commands.Bot(command_prefix="!", intents= discord.Intents.all())

#defined vibes to be referenced later in the vibe function
vibes = ['Chill vibes','Good vibes','Bad vibes','Wack','Meh','This is fine','Could be better','Could be worse']

#defined responses for the magic 8 ball function
magic_ball_responses = ('It is certain', 'It is decidedly so', 'Without a doubt', 'Yes, definitely',
 'You may rely on it', 'As I see it, yes', 'Most likely', 'Outlook good',
 'Signs point to yes', 'Yes', 'Reply hazy, try again', 'Ask again later',
 'Better not tell you now', 'Cannot predict now', 'Concentrate and ask again',
 "Don't bet on it", 'My reply is no', 'My sources say no', 'Outlook not so good',
 'Very doubtful')

#defined rock paper scissors winning combos
winning_combo = {
    'rock':'scissors',
    'paper':'rock',
    'scissors':'paper'
}
#initiates the bot
@bot.event
async def on_ready():
    print('The Oracle has awoken')
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(e)

#gets a random vibe from the above list
@bot.tree.command(name='vibe',description='Vibe check')
async def vibe(interaction: discord.Interaction):
    vibes_length = len(vibes)
    choice_index = random.randint(0,vibes_length-1)
    embed = discord.Embed(title='Vibe check', description=vibes[choice_index], color=discord.Color.random())
    await interaction.response.send_message(embed=embed)

#this will run the fortune - cowsay command in local terminal and send the output as a message
@bot.tree.command(name='fortune',description='Get your fortune!')
async def fortune(interaction: discord.Interaction):
    cowTypes = getoutput('/usr/games/cowsay -l')[37:]
    cowTypes = cowTypes.split()  # split into cowsay animals
    typechoice = cowTypes[randrange(0, len(cowTypes), 1)]
    # Use our choice to generate a cowsay
    msg = getoutput('/usr/games/fortune | /usr/games/cowsay -f {}'.format(typechoice))
    # Image generation: calculate length and width of image and instantiate
    msgFont = ImageFont.truetype("UbuntuMono-Regular.ttf", 12)
    msgDim = msgFont.getsize_multiline(msg)
    msgImg = Image.new('RGB', (ceil(
        msgDim[0] + 0.1*msgDim[0]), ceil(msgDim[1] + 0.1*msgDim[1])), (54, 57, 62, 0))
    msgDraw = ImageDraw.Draw(msgImg)
    msgDraw.text((16, 0), msg, fill=(255, 255, 255, 255), font=msgFont)
    msgImg.save('/tmp/fortune.png')
    embed=discord.Embed(title='The Oracle Says:', color=discord.Color.random())
    file = discord.File('/tmp/fortune.png', filename='fortune.png')
    embed.set_image(url="attachment://fortune.png")
    await interaction.response.send_message(embed=embed, file=file)

#this will run the magic 8 ball
@bot.tree.command(name='8ball',description='Consult the magic 8 ball!')
async def magic_ball(interaction: discord.Interaction):
    magic_answer = random.choice(magic_ball_responses)
    embed=discord.Embed(title='The Oracle Says: ', description=magic_answer, color=discord.Color.random())
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name='flip',description='Flip a coin heads or tails!')
async def coin_flip(interaction: discord.Interaction):
    results = {
        0 : 'Heads',
        1 : 'Tails'
    }
    index = randrange(0,2)
    flip_result = results[index]
    embed=discord.Embed(title='Coinflip!', description=f'You flipped: {flip_result}!', color=discord.Color.random())
    await interaction.response.send_message(embed=embed)


#help command to list current functions
@bot.tree.command(name='help',description='Display the help message')
async def help_func(interaction: discord.Interaction):
    embed=discord.Embed (title='Help!', description='''I can only do a few things at the moment:
/fortune:  Will run the cowsay fortunes command!
/flip:  Will flip a coin heads or tails Style!
/8ball:  Will give a magic 8ball response!
/rps: </rps @anyone> in the server and reply to the dm with Rock Paper or Scissors
/aidraw prompt to have replicate generate an image
/askai question to get a response from chatGPT''', color=discord.Color.random())
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name='rps',description='@ another user and reply to the bots DM to play!')
async def rps(interaction: discord.Interaction, secondplayer:discord.Member):

    #define styles for the embeds
    embed_rps=discord.Embed(title='The Oracle says:',color=discord.Color.random())
    embed_rps.set_thumbnail(url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUNjdD-Vfq-_MZpu-KZpUdqmiXmqV4FcEr_lLmuCyyYsdA7r_MHhPh9dLVwSA2GQa9Bvg&usqp=CAU')
    embed_dm=discord.Embed(title='The Oracle says:',color=discord.Color.random())
    embed_dm.set_thumbnail(url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUNjdD-Vfq-_MZpu-KZpUdqmiXmqV4FcEr_lLmuCyyYsdA7r_MHhPh9dLVwSA2GQa9Bvg&usqp=CAU')
    embed_dm.add_field(name='Rock Paper Scissors!',value='Send Rock Paper or Scissors into chat below!')

    try:
        #define players and gather user id's
        player1_name = interaction.user
        player1 = interaction.user.id
        player2 = secondplayer.id


        #checks if both players are from the same instance
        async def check_player1(message):
                return  message.author.id == player1 and isinstance(message.channel, discord.DMChannel)
        async def check_player2(message):
                return message.author.id == player2 and isinstance(message.channel, discord.DMChannel)

        #this entire function is cancer and I don't know how to simplify
        async def rps_game():
            #send first player inital message and stores choice
            player1_message = await player1_name.create_dm()
            await player1_message.send(embed=embed_dm)
            player1_choice = await bot.wait_for('message', check=check_player1)
            player1_compare = player1_choice.content.lower()

            #these while loops ensure the key being passed into the winner check is a valid key in the list
            while player1_compare != 'rock' and player1_compare != 'paper' and player1_compare != 'scissors':
                await player1_message.send('Choose again')
                player1_choice = await bot.wait_for('message', check=check_player1)
                player1_compare = player1_choice.content.lower()
                if player1_compare == 'rock' or player1_compare == 'paper' or player1_compare == 'scissors':
                    break

            #send second player initial message and stores choice
            player2_message = await secondplayer.create_dm()
            await player2_message.send(embed=embed_dm)
            player2_choice = await bot.wait_for('message', check=check_player2)
            player2_compare = player2_choice.content.lower()

            #these while loops ensure the key being passed into the winner check is a valid key in the list
            while player2_compare != 'rock' and player2_compare != 'paper' and player2_compare != 'scissors':
                await player2_message.send('Choose again')
                player2_choice = await bot.wait_for('message', check=check_player2)
                player2_compare = player2_choice.content.lower()
                if player2_compare == 'rock' or player2_compare == 'paper' or player2_compare == 'scissors':
                    break

            #the actual game logic
            if player1_compare == player2_compare:
                embed_rps.add_field(name='Draw!', value="\u200b",inline=False)

            elif player1_compare == winning_combo[player2_compare]:
                embed_rps.add_field(name=f'{secondplayer} Won!', value="\u200b",inline=False)

            elif player2_compare == winning_combo[player1_compare]:
                embed_rps.add_field(name=f'{player1_name} Won!', value="\u200b",inline=False)

            else:
                print('error')
            embed_rps.add_field(name=f'{player1_name}',value=f"<@{player1}>\nChose: {player1_compare}")
            embed_rps.add_field(name=f'{secondplayer}',value=f"<@{player2}>\nChose: {player2_compare}")
            await interaction.channel.send(embed=embed_rps)

        await rps_game()

    except Exception as e:
        print(e)



@bot.tree.command(name='askai',description='Ask a question to ChatGPT!')
async def ask_ai(interaction: discord.Interaction, question:str):
    channel = interaction.channel
    msg_embed = discord.Embed(title='GPT4 Says:',description=f"{question}\n> Generating...")
    msg = await interaction.response.send_message(embed=msg_embed)
    completion = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role":"user","content":question}]
    )
    response = completion.choices[0].message.content
    embed = discord.Embed(title='ChatGPT Says:', description=response)
    embed.set_footer(text=question)
    await interaction.edit_original_response(embed=embed)


@bot.tree.command(name='aidraw',description='Provide a prompt to Dall-e-3 to generate an image!')
async def ai_draw(interaction:discord.Interaction, prompt:str):
    try:
        channel = interaction.channel
        msg_embed = discord.Embed(title='Dall-e is drawing:', description=f"\n{prompt}\n> Generating...")
        msg = await interaction.response.send_message(embed=msg_embed)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        response_embed = discord.Embed(title='Your masterpiece',description=interaction.user.mention)
        response_embed.set_image(url=image_url)
        response_embed.set_footer(text=prompt)
        await interaction.edit_original_response(embed=response_embed)
    except Exception as e:
        if str(e) == 'Your request was rejected as a result of our safety system. Your prompt may contain text that is not allowed by our safety system.':
            await channel.send(interaction.user.mention+'\n'+str(e))

bot.run(token)
