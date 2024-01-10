from openai import OpenAI
import maricon
client = OpenAI(api_key=maricon.gptkey)
import personality
import asyncio
import random


# Initialize a dictionary to store conversation history for each user
user_conversations = {}
cm_chat_conversations = {}

#for translation module, this can be changed to any language
translate_language = 'spanish'


async def generate_text_with_timeout(prompt, user_id):
    try:
        return await asyncio.wait_for(generate_text(prompt, user_id), timeout=10)
    except asyncio.TimeoutError:
        return "obama"

async def generate_text(prompt,user_id,personality_context=personality.malik):
    # Check if the user already has a conversation history
    if user_id not in user_conversations:
        user_conversations[user_id] = ""

    # Update the conversation history with the new message
    user_conversations[user_id] += f"\nUser {user_id}: {prompt}"

    #full_prompt = f"{personality_context} \n {user_conversations[user_id]}"
    full_prompt = [{"role": "user", "content": f"{personality_context.prompt} \n {user_conversations[user_id]}"}]

    response = client.chat.completions.create(model="gpt-4-1106-preview",
    max_tokens=200,
    temperature=0.8,
    messages = full_prompt)

    print(response)
    
    generated_text = response.choices[0].message.content.strip()

    user_conversations[user_id] += f"\nAI: {generated_text}"

    print(f'debug + {user_conversations[user_id]}')

    print('text before replace: ' + generated_text)
    
    if ":" in generated_text:
        generated_text = generated_text[generated_text.find(":")+1:]
    
    generated_text = str(generated_text.replace('AI:',''))

    # clear user conversation if it has more than 15 ':'s
    # save money!
    if user_conversations[user_id].count(':') > 15:
        user_conversations[user_id] = ""

    return generated_text

    
  
async def spanish_translation(prompt):
    language_prompt = f'Translate chatroom message from english to {translate_language}, keep similar grammar/formality:'
    try:
        return await asyncio.wait_for(generate_text_gpt_spanish(language_prompt + '\n' +  prompt), timeout=15)
    except asyncio.TimeoutError:
        return "obama"
    
async def generate_text_with_timeout_gpt(prompt):
    try:
        return await asyncio.wait_for(generate_text_gpt(prompt), timeout=15)
    except asyncio.TimeoutError:
        return "obama"
    
#basic gpt
async def generate_text_gpt(prompt):

    prompt = prompt.replace('!gpt','')

    full_prompt = [
        {"role": "system", "content": "you are Ari, you are posting in a discord channel. you will respond with short informal messages. you will not refer to yourself as an AI."},
        {"role": "user", "content": f"{prompt}"}
        ]

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    max_tokens=1200,
    temperature=.8,
    messages = full_prompt)

    #print(response)
    
    generated_text = response.choices[0].message.content.strip()

    #force lowercase
    generated_text = generated_text.lower()

    # replace words in generated text with gato slang
    for word in generated_text.split(' '):
        if word in gato_slang:
            generated_text = generated_text.replace(word, gato_slang[word])


    return generated_text

gato_slang = {
    'guitar': 'juitar',
    'guitars': 'juitars',
    'album': 'albumin',
    'alcohol' : 'algoman',
    'alcoholic' : 'algomanic',
    'smoking' : 'smogging',
    'smokin' : 'smoggin',
    'y\'all' : 'yall',
    'barinade' : 'bari',
    'as an ai language model' : '',
}

async def generate_text_gpt_spanish(prompt):

    full_prompt = [
        {"role": "user", "content": f"{prompt}"}
        ]

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    max_tokens=1200,
    temperature=.8,
    messages = full_prompt)

    #print(response)
    
    generated_text = response.choices[0].message.content.strip()

    #force lowercase
    generated_text = generated_text.lower()

    return generated_text

async def ai_experimental(prompt, gmodel="gpt-3.5-turbo", prompt_addition=False):

    #in prompt we receive a list of messages. each item is user: message, convert it into prompt. if user is ari, then role is system'

    if prompt_addition:
        rprole = prompt_addition
    else:
        rprole = 'ari'

    promptcontainer = [{"role": "system", "content": f"you are role playing as {rprole}, you are posting in a discord channel. short and informal chat style messages. you will not refer to yourself as an AI. do not use emojis or hashtags. do not end your sentence with a question"}]

    for message in prompt:
        #parse user
        user = message.split(':')[0]
        #message is everything after the first colon as there may be multiple colons in the message
        #message = message.split(':',1)[1]
        if user == 'ari':
            role = 'assistant'
        else:
            role = 'user'
        promptcontainer.append({"role": role, "content": f"{message}"})
    print(promptcontainer)


    full_prompt = [
        {"role": "user", "content": f"{prompt}"}
        ]

    response = client.chat.completions.create(model=gmodel,
    max_tokens = 800,
    temperature=.8,
    messages = promptcontainer)

    print(response)
    generated_text = response.choices[0].message.content.strip()

    #force lowercase
    generated_text = generated_text.lower()

    return generated_text
