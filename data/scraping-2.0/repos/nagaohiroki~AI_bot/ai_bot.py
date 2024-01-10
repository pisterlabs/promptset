import discord
import os
import openai


intents = discord.Intents.all()
client = discord.Client(intents=intents)
chat_context = {}
chat_messages = {}


@client.event
async def on_ready():
    openai.api_key = os.environ['OPENAI_API_KEY']
    print('AI bot login')
            


@client.event
async def on_message(message):
    if message.author.bot:
        print('bot echo')
        return
    if message.channel.name != 'ai':
       if client.user not in message.mentions:
            print(f'no mention. at: {message.channel}')
            return
    text = ''
    try:
        img = '/img'
        role = '/role'
        if img in message.clean_content:
            text = create_ai_image(message.clean_content.replace(img, ''))
        elif role in message.clean_content:
            create_role(message)
            return
        else:
            text = create_gpt_chat(message)
    except Exception as e:
        text = str(e)
        global chat_context
        global chat_messages
        chat_context = {}
        chat_messages = {}
    await message.channel.send(text)


def create_ai_chat(message):
    global chat_context
    context_text = chat_context.get(message.guild.id)
    if not context_text:
        context_text = ''
    prompt = f'{context_text}{message.author.name}:{message.clean_content}\nAI:'
    # clamp max_token avoid overflow error.
    max_token = 2048
    prompt = prompt[-max_token:]
    ai_text = create_ai_text(prompt)
    chat_context[message.guild.id] = f'{prompt}{ai_text}\n'
    return ai_text


def create_ai_text(prompt):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=1024,
        temperature=0.7)
    print(response)
    return response.choices[0]['text']


def create_ai_image(prompt):
    response = openai.Image.create(prompt=prompt, n=1, size="512x512")
    print(response)
    return response.data[0]['url']


def create_role(message):
    id = message.guild.id
    messages = chat_messages.get(id)
    new_role = message.clean_content.replace('/role', '')
    role = {'role': 'system', 'content': new_role}
    if not messages:
        messages = [role]
        chat_messages[id] = messages
        return
    messages.append(role)


def create_gpt_chat(message):
    id = message.guild.id
    messages = chat_messages.get(id)
    if not messages:
        messages = [{'role': 'system', 'content': 'discord bot'}]
        chat_messages[id] = messages
    prompt = f'{message.author.name}:{message.clean_content}'
    messages.append({'role': 'user', 'content': prompt})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = messages)
    new_message = response.choices[0].message.content
    messages.append({'role': 'assistant', 'content': new_message})
    if context_size(messages) > 4000:
        messages = [{'role': 'system', 'content': 'discord bot'}]
    return new_message.replace('Bot:', '')


def context_size(messages):
    size = 0
    for message in messages:
        size += len(message['content'])
    return size


client.run(os.environ['AI_BOT_TOKEN'])
