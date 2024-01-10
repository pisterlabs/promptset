import os
import pickle
import openai
from pydub import AudioSegment


## Datos
server_settings = {}
channel_ids = {}
GPT_Model = "gpt-3.5-turbo"

# Obtener la ruta del directorio actual
dir_path = os.path.dirname(os.path.realpath(__file__))

# Concatenar el nombre del archivo de texto al final de la ruta
server_settings_file_path = os.path.join(dir_path, "config\\server_settings.pickle")

channel_ids_file_path = os.path.join(dir_path, "config\\channel_ids.pickle")

# Concatenar el nombre del archivo de audio al final de la ruta
audio_file_path = os.path.join(dir_path, "transcriptions\\voice-message.mp3")


def validServer(server_id):
    return server_id in server_settings


# cambiar luego por channel_ids
def validChannel(message):
    return message.channel.id in channel_ids


def featureEnabled(server_id):
    return server_settings[server_id]["feature_enabled"]


# Funcion para obtener la respuesta de OpenAI a partir de un prompt
def get_completion(prompt, model=GPT_Model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
        max_tokens=400,  # this is the maximum number of tokens that the model will generate
    )
    return response.choices[0].message["content"]


# Funcion para obtener una imagen generada por OpenAI a partir de un prompt
def get_image(prompt):
    response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
    image_url = response["data"][0]["url"]
    return image_url

def get_translation(target_message, target_language="en", model=GPT_Model):
    prompt = f"Translate the following message to {target_language}:\n{target_message}"
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
        max_tokens=400,  # this is the maximum number of tokens that the model will generate
    )
    return response.choices[0].message["content"]


def isVoiceMessage(message):
    attachment = message.attachments[0]
    file_extension = attachment.filename.split(".")[-1]
    if file_extension in ["ogg"]:
        return True


async def save_audio_file(message):
    for attachment in message.attachments:
        # Save the audio file to disk
        filename = attachment.filename
        print("Saving audio file to:", audio_file_path)

        try:
            with open(audio_file_path, "wb") as f:
                await attachment.save(f)
                f.close()
        except FileNotFoundError:
            # Si no existe el directorio, lo crea
            if not os.path.exists(os.path.dirname(audio_file_path)):
                os.makedirs(os.path.dirname(audio_file_path))
            with open(audio_file_path, "wb") as f:
                await attachment.save(f)
                f.close()


def format_to_mp3():
    # Load the audio file
    audio_file = AudioSegment.from_file(audio_file_path)

    # Export the audio file in the MP3 format
    audio_file.export(audio_file_path, format="mp3")


# Funcion para transcribir audio a texto, utilizando el modelo "whisper-1" de OpenAI
def transcribe_audio():
    format_to_mp3()

    audio_file = open(audio_file_path, "rb")
    transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file)
    return transcript.text


# Carga los server settings, atrapando la excepcion en caso de que no exista el archivo
def load_server_settings():
    try:
        with open(server_settings_file_path, "rb") as f:
            global server_settings
            server_settings = pickle.load(f)
            f.close()
    except FileNotFoundError:
        # Si el directorio no existe, lo crea
        if not os.path.exists(os.path.dirname(server_settings_file_path)):
            os.makedirs(os.path.dirname(server_settings_file_path))
        server_settings = {}
        with open(server_settings_file_path, "wb") as f:
            pickle.dump(server_settings, f)
            f.close()


def toggle_feature(server_id):
    # Si el servidor no esta en la lista de servidores, lo agrega
    if server_id not in server_settings:
        server_settings[server_id] = {}

    # Cambia el estado de la caracteristica
    server_settings[server_id]["feature_enabled"] = not server_settings[server_id].get(
        "feature_enabled", False
    )

    # Guarda los cambios en el archivo
    with open(server_settings_file_path, "wb") as f:
        pickle.dump(server_settings, f)
        f.close()


def load_channel_ids():
    try:
        with open(channel_ids_file_path, "rb") as f:
            global channel_ids
            channel_ids = pickle.load(f)
            f.close()
    except FileNotFoundError:
        # Si el directorio no existe, lo crea
        if not os.path.exists(os.path.dirname(channel_ids_file_path)):
            os.makedirs(os.path.dirname(channel_ids_file_path))
        channel_ids = {}
        with open(channel_ids_file_path, "wb") as f:
            pickle.dump(channel_ids, f)
            f.close()


def add_channel(channel_id):
    if channel_id not in channel_ids:
        channel_ids[channel_id] = {}

    with open(channel_ids_file_path, "wb") as f:
        pickle.dump(channel_ids, f)
        f.close()


def remove_channel(channel_id):
    if channel_id in channel_ids:
        del channel_ids[channel_id]
        with open(channel_ids_file_path, "wb") as f:
            pickle.dump(channel_ids, f)
            f.close()


# Function to set the model to use. The model can be "3", "4" or "4 Vision". Defaults to gpt-3.5-turbo everytime the bot gets reloaded, to avoid wasting money.
def set_model(model):
    global GPT_Model

    # If the model is equal to "3", set the model to "gpt-3.5-turbo". If the model is equal to 4, set the model to "gpt-4-1106-preview". If the model is equal to "4 Vision", set the model to "gpt-4-vision-preview".
    if model == "3":
        GPT_Model = "gpt-3.5-turbo"
    elif model == "4":
        GPT_Model = "gpt-4-1106-preview"
    elif model == "4 Vision":
        GPT_Model = "gpt-4-vision-preview"