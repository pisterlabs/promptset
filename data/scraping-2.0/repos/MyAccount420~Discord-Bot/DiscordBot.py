#python3
#bots name is Brainy
import openai
import discord
import os

openai.api_key = '' # openai API key <<< Sign up for open and enter API key

TOKEN = '' #discord bot token <<<< discord bot token here

client = discord.Client()
count = 0
prompt = ""
names = []
custom_prompt = ''
temperature = 0.75
max_tokens = 200
top_p = 1
frequency_penalty = 1.75
presence_penalty = 1.23
promptbackup = ''

file = 'DiscordBot.py' #name if this file
discordchannel = 'discordbot' #name of discord channel <<<< Change this to the nameo of the channel you want to bot to interact with

def word_count(string):
    return (len(string.strip().split(" ")))


def get_response(user_message, username, names, custom_prompt, temperature, max_tokens, top_p, frequency_penalty,
                 presence_penalty):
    global count
    global prompt

    # Default prompt initial
    if count == 0 and prompt == '':
        # set prompt value to default message
        prompt = "This is a discord conversation with a bot named Brainy\n<<conversation>>\n" + str(username) \
                 + ": " + \
                 str(user_message) + "\nBrainy McBrainson:"
        print("prompt == None")
    else:
        #continue conversation
        prompt = str(prompt) + '\n' + str(username) + ": " + str(user_message) + "\nBrainy McBrainson:"
        print("continue with conversation")
        if custom_prompt != '':
            #initial set of custom prompt
            custom_prompt =str(custom_prompt) + '\n' + str(username) + ": " + str(user_message) + "\nBrainy McBrainson:"
            prompt = str(custom_prompt)

            print(f"prompt in custom prompt set == {prompt}")



    length = word_count(prompt)
    if length > max_tokens:
        #shorten prompt to 500 words(deleting first word)
        prompt = prompt.split(" ", max_tokens)[1:]
        prompt = " ".join(prompt)
        print(f"prompt shortened to {max_tokens} words == {prompt}")

    count += 1
    print("\n\nstart prompt")
    print(prompt)
    print('end prompt\n\n')
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=names
    )
    output = response['choices'][0]['text']
    prompt = prompt + output

    # call the openai api
    return output


# a fucntion that takes in a string and returns a string


@client.event
async def on_ready():
    print('Logged in as {0}'.format(client.user))


@client.event
async def on_message(message):
    global names
    global custom_prompt
    global temperature
    global max_tokens
    global top_p
    global frequency_penalty
    global presence_penalty
    global discordchannel
    global prompt
    global promptbackup
    print("\n\n        -New Message- \n\n")

    username = str(message.author).split('#')[0]
    user_message = str(message.content)
    channel = str(message.channel.name)
    print(f'{username}: {user_message} ({channel})')

    if message.channel.name == discordchannel:
        human_name = str(message.author).split('#')[0]
        # add a ':' to the end of the human_name
        human_name = human_name + ':'
        custom_prompt = ''

        if message.content == '!restart':
            #send message saying "restarting ..."
            await message.channel.send('Restarting ...')
            os.system(f'python3 {file}')
            return

        if message.content == '!startover':
            if promptbackup != '':
                await message.channel.send('Starting over ...')
                return
            else:
                await message.channel.send('No !prompt was set')
            return


        if message.content == '!help':
            await message.channel.send(f"<<Common settings>>\n\n!expert - use !expert+topic to make the bot an expert on whatever topic you give it \n\n !prompt - Sets the personality of the bot \n\n!restart - restarts the bot forgetting all settings and custom prompt \n\n<<Advanced Options>>\n\n!show-prompt - Shows you the conversations history (debugging only)\n\n !settings shows you the current OPENAI settings \n\n")
            return
        if message.content == '!show-prompt':
            await message.channel.send(prompt)
            return

        if message.content == '!settings':
            await message.channel.send(f" !temp= sets the temperature aka Randomness of responses (Max=2)\nCurrent temperature={temperature} \n\n !max= sets the max_tokens aka max number of words in response(Max=3000)\nCurrent max_tokens={max_tokens} \n\n !top= sets the top_p, honestly no idea what this setting is lol(Max=1)\nCurrent top_p{top_p} \n\n !freq= sets the frequency_penalty, the higher the value the less likley the bot is to reapeat himself(Max=2)\nCurrent frequency_penalty={frequency_penalty} \n\n !pres= Presence penalty sets the, the higher the more likley the bot is to come up with new topics(Max=2)\nCurrent Presence penalty={presence_penalty}")
            return

        #set the openai api settings
        if message.content.startswith('!temp='):
            temperature = float(message.content.split('!temp=')[1])
            print(f"temperature is {temperature}")
            await message.channel.send("Temperature set")
            return temperature
        if message.content.startswith('!max='):
            max_tokens = float(message.content.split('!max=')[1])
            print(f"max_tokens is {max_tokens}")
            await message.channel.send("max_tokens set")
            return max_tokens
        if message.content.startswith('!top='):
            top_p = float(message.content.split('!top=')[1])
            print(f"top_p is {top_p}")
            await message.channel.send("top_p set")
            return top_p
        if message.content.startswith('!freq='):
            frequency_penalty = float(message.content.split('!freq=')[1])
            print(f"frequency_penalty is {frequency_penalty}")
            await message.channel.send("frequency_penalty set")
            return frequency_penalty
        if message.content.startswith('!pres='):
            presence_penalty = float(message.content.split('!pres=')[1])
            print(f"presence_penalty is {presence_penalty}")
            await message.channel.send("presence_penalty set")
            return presence_penalty



        # if the message starts with !prompt
        if message.content.startswith('!prompt'):
            # set prompt to the message after the !prompt
            custom_prompt = message.content.split('!prompt')[1]
            prompt = custom_prompt

            promptbackup = custom_prompt


            # send a message to the channel that says "prompt set"
            await message.channel.send("Prompt set")

            return

        # if message starts with !expert
        if message.content.startswith('!expert'):
          custom_prompt = message.content.split('!expert')[1]
          custom_prompt = f"Brainy is an expert in the field of {custom_prompt} his knowledge is broad and deep, he is here to help answer an questions you may have on the topic in a clear and truthful manner. His information is practical and actionable.  \n <<Conversation Starts>>"
          prompt = custom_prompt

          promptbackup = custom_prompt

          # send a message to the channel that says "prompt set"
          await message.channel.send("Prompt set")

          return

        if message.author == client.user:
            return

        # if the human_name is not in the names list
        if human_name not in names:
            # add name to the list
            names.append(human_name)



            # call a function that takes the user message and returns a string
        response = get_response(user_message, username, names, custom_prompt, temperature, max_tokens, top_p,
                                frequency_penalty, presence_penalty)
        await message.channel.send(response)


client.run(TOKEN)
