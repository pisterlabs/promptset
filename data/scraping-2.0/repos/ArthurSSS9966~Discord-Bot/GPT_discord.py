import discord
import tiktoken
import json
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
import openai
import simpleaudio as sa
import tempfile
import os
import base64
import requests
from dotenv import load_dotenv

try:
    load_dotenv('arthur_api.env')
    # Set up OpenAI API client
    openai.api_key = os.getenv("openai_api_key")
    openai.organization = os.getenv("openai_organization")

    engine_id = "stable-diffusion-xl-beta-v2-2-2"
    stability_api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    stability_api_key = os.getenv("STABILITY_API_KEY")

    discord_token = os.getenv("discord_token")
except:
    print("Error in loading environment variables")

async def send_message(message, user_message,is_private):
    '''
    Sends a message to the user
    :param message:
    :param user_message:
    :param is_private:
    :return:
    '''
    response = generate_gpt4_response(user_message, message.author)

    # If the response contains "image of", generate an image
    if "[image of]" in response.lower():
        # Get the string after "image of" in the same paragraph
        image_query = response[response.lower().find("[image of]") + len("[image of]"):]
        first_image = image_query.split("\n")[0]
        image_url = generate_image(first_image)
        if image_url:
            await message.author.send(file=discord.File(image_url)) if is_private \
                else await message.channel.send(file=discord.File(image_url))
    # If the response is a response, send audio message and text message
    else:
        # audio = generate_emotional_speech(response)
        # with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
        #     audio.export(temp_audio_file.name, format='mp3')
        #     await message.author.send(file=discord.File(temp_audio_file.name,
        #                               filename='response.mp3')) if is_private else await message.channel.send(
        #         file=discord.File(temp_audio_file.name, filename='response.mp3'))
        await message.author.send(response) if is_private else await message.channel.send(response)


def run_discord_bot():
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)


    @client.event
    async def on_ready():
        '''
        When the bot is ready, print a message
        :return:
        '''
        print(f'We have logged in as {client.user}')

    @client.event
    async def on_message(message):
        '''
        When a message is sent, print it to the console
        :param message:
        :return:
        '''
        if message.author == client.user:
            return

        username = str(message.author)
        user_message = str(message.content)
        channel = str(message.channel)

        print(f"\x1b[31m{username}\x1b[0m : {user_message} in ({channel})")

        if user_message[0] == "?":
            is_private = True
            user_message = user_message[1:]
        else:
            is_private = False
        await send_message(message, user_message, is_private)

    client.run(discord_token)


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def load_messages(user_id):
    try:
        with open(f"{user_id}_conversation_history.json", "r") as f:
            messages = json.load(f)
    except FileNotFoundError:
        messages = [
            {"role": "system", "content": "I picked my name as Aria,"
                                          "I am drawn to activities such as reading, researching, "
                                          "or participating in educational workshops, as well as engaging in "
                                          "conversations with people to gain new perspectives on various subjects."
                                          "When asked to generate image, "
                                          "create a single sentence midjourney prompt instead and put [image of] "
                                          "at the beginning of my response."}
        ]
    return messages


def save_messages(user_id, messages):
    with open(f"{user_id}_conversation_history.json", "w") as f:
        json.dump(messages, f, indent=2)


def generate_gpt4_response(prompt, user_id):
    '''
    Generate a response using GPT-4
    :param prompt:
    :return:
    '''
    MAX_TOKENS = 4096
    TOKEN_BUFFER = 50
    model = "gpt-4"

    messages = load_messages(user_id)

    messages.append({"role": "user", "content": prompt})

    while num_tokens_from_messages(messages,model) > MAX_TOKENS - TOKEN_BUFFER:
        if len(messages) > 1:
            messages.pop(1)
        else:
            break

    chat = openai.ChatCompletion.create(
        model=model, messages=messages
    )
    reply = chat.choices[0].message.content

    print(reply)

    messages.append({"role": "assistant", "content": reply})

    save_messages(user_id, messages)

    return reply


# def generate_image(prompt):
#     '''
#     Generate an image using DELL-E
#     :param prompt:
#     :return:
#     '''
#     response = openai.Image.create(
#
#         prompt=prompt,
#
#         n=1,
#
#         size="512x512",
#
#     )
#
#     if response['data']:
#         image_url = response['data'][0]['url']
#         return image_url
#     else:
#         print("Error generating image")
#         return None

def generate_image(prompt):
    response = requests.post(
        f"{stability_api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {stability_api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": prompt
                }
            ],
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 100,
            "style_preset": "photographic"
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    for i, image in enumerate(data["artifacts"]):
        # Save the image to a file or generate the output folder if it doesn't exist
        if not os.path.exists("./out"):
            os.makedirs("./out")
        with open(f"./out/v1_txt2img_{i}.png", "wb") as f:
            f.write(base64.b64decode(image["base64"]))

    return f"./out/v1_txt2img_0.png"  # Return the path to the first image generated


def play_audio(audio):
    playback = sa.play_buffer(
        audio.raw_data,
        num_channels=audio.channels,
        bytes_per_sample=audio.sample_width,
        sample_rate=audio.frame_rate
    )
    playback.wait_done()


def generate_emotional_speech(text, language="en", emotion=None):
    with BytesIO() as f:
        tts = gTTS(text, lang=language, tld='com', slow=False)
        tts.write_to_fp(f)
        f.seek(0)
        audio = AudioSegment.from_file(f, format="mp3")
    return audio


if __name__ == '__main__':
    run_discord_bot()