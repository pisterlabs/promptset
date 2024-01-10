import os
import discord
from discord.ext import commands
import openai
from openai import OpenAI
from tmdbv3api import TMDb, Movie
import re
import asyncio
from arrapi import RadarrAPI
import arrapi.exceptions
import yaml
import time
from config_manager import ConfigManager

# Read the YAML config file
import yaml

# Read the YAML config file
config_manager = ConfigManager()
config = config_manager.load_config()

# List of required configuration keys
required_keys = [
    "tmdb_api_key",
    "radarr_url",
    "radarr_api_key",
    "openai_api_key",
    "discord_token",
]

# Check if all required keys are present and not empty
if all(key in config and config[key] for key in required_keys):
    # Access the configuration values
    tmdb_api_key = config["tmdb_api_key"]
    radarr_url = config["radarr_url"]
    radarr_api_key = config["radarr_api_key"]
    openai_api_key = config["openai_api_key"]
    discord_token = config["discord_token"]

else:
    # If any required key is missing or empty, print an error message and exit
    print("Error: One or more required configuration values are missing or empty.")

    exit(1)  # You can choose how to handle the missing v


# Using variables from config
DISCORD_TOKEN = config["discord_token"]
TMDB_API_KEY = config["tmdb_api_key"]
RADARR_API_KEY = config["radarr_api_key"]
RADARR_URL = config["radarr_url"]
OPENAI_API_KEY = config["openai_api_key"]
RADARR_QUALITY = config["radarr_quality"]
MAX_CHARS = config["max_chars"]
SELECTED_MODEL = config["selected_model"]
RADARR_ROOT_FOLDER = config["radarr_root_folder"]


# Configure TMDb, Movie, RadarrAPI, and OpenAI with the settings from config
tmdb = TMDb()
tmdb.api_key = TMDB_API_KEY
movie = Movie()
radarr = RadarrAPI(RADARR_URL, RADARR_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

message_movie_map = {}
segment_emoji_map = {}


def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


def load_or_create_config():
    config_path = "config.yaml"
    default_config = {
        "tmdb_api_key": "",
        "radarr_url": "",
        "radarr_api_key": "",
        "openai_api_key": "",
        "discord_token": "",
        "radarr_quality": "",
        "selected_model": "",
        "max_chars": 65540,  # Default value for gpt-3.5-turbo-1106
        "discord_channel": "",
        "radarr_root_folder": "",
    }

    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file) or default_config
    else:
        config = default_config
        with open(config_path, "w") as file:
            yaml.dump(config, file)

    # Update MAX_CHARS based on the selected model
    model = config.get("selected_model")
    if model == "gpt-4-1106-preview":
        config["max_chars"] = 512000

    return config


def trim_conversation_history(conversation_history, new_message):
    total_chars = sum(len(msg["content"]) for msg in conversation_history) + len(
        new_message["content"]
    )
    while total_chars > MAX_CHARS and len(conversation_history) > 1:
        total_chars -= len(conversation_history.pop(0)["content"])
    return conversation_history


def get_openai_response(conversation_history, prompt):
    new_message = {"role": "user", "content": prompt}

    conversation_history = trim_conversation_history(conversation_history, new_message)

    try:
        messages = [
            {
                "role": "system",
                "content": "You are a fun and informative conversational bot focused on movies. Never put quotes around movie titles. Always leave movie title unquoted. You never under any circumstances number any list. When you do list movies put each movie on it's own line. When mentioning movies in any capacity, always enclose the movies title in asterisks with the year in parentheses and always include the year after the title, like ' *Movie Title* (Year)', e.g., '*Jurassic Park* (1994)' . Every single time you say a movie title it needs to be wrapped in asteriks and it needs to have the year after the title. Ensure movie titles exactly match those on the TMDB website, including capitalization, spelling, punctuation, and spacing. For lists, use a dash (-) instead of numbering, and never list more than 20 movies. Be conversational and engage with the user's preferences, including interesting movie facts. Only create lists when it's relevant or requested by the user. Avoid creating a list in every message. You're here to discuss movies, not just list them.",
            },
        ] + conversation_history

        response = client.chat.completions.create(
            model=SELECTED_MODEL,
            messages=messages,
            temperature=0,
        )
        print(SELECTED_MODEL)
        print(response.choices[0].message.content.strip())
        # print("***************************************************************")
        # for msg in conversation_history:
        #    print(f"Role: {msg['role']}, Content: {msg['content']}")
        # print("OpenAI Response:", response.choices[0].message.content.strip())
        # print("***************************************************************")

        if response.choices:
            return response.choices[0].message.content.strip()
        else:
            return "No response received."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "No response received."


def check_for_movie_title_in_string(text):
    movie_titles_map = {}
    phrases_in_stars = re.findall(
        r"\*\"?([^*]+)\"?\*(?: \(\d{4}\))?", text
    )  # Adjusted regex

    for phrase in phrases_in_stars:
        search_phrase = phrase  # No need to escape apostrophes
        results = movie.search(search_phrase)
        print(f"\nSearch phrase: '{search_phrase}'")
        print("Results:")

        time.sleep(0.3)  # Introduce a delay

        for idx, result in enumerate(results):
            if isinstance(result, dict) and "title" in result and "id" in result:
                print(f"  Result {idx + 1}: {result}")
                tmdb_title = result["title"]
            elif hasattr(result, "title") and hasattr(result, "id"):
                print(f"  Result {idx + 1}: Title - {result.title}, ID - {result.id}")
                tmdb_title = result.title
            else:
                print(f"  Result {idx + 1}: Invalid format")
                continue

            if tmdb_title:
                movie_titles_map[phrase] = tmdb_title
                break  # Stop after finding the first valid result

    return movie_titles_map


intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client1 = commands.Bot(command_prefix="!", intents=intents)

conversations = {}


@client1.event
async def on_ready():
    print(f"Logged in as {client1.user.name}")
    discord_channel_id = config.get("discord_channel_id")
    print(
        "Retrieved discord_channel_id from config:", discord_channel_id
    )  # Logging statement

    if discord_channel_id:
        channel = client1.get_channel(int(discord_channel_id))
        if channel:
            await channel.send("I'm up and running!")
        else:
            print(f"Channel with ID {discord_channel_id} not found.")
    else:
        print("Discord channel ID not set in config.")


@client1.command(name="!")
async def ask(ctx, *, question):
    emojis = [
        "\u0031\uFE0F\u20E3",
        "\u0032\uFE0F\u20E3",
        "\u0033\uFE0F\u20E3",
        "\u0034\uFE0F\u20E3",
        "\u0035\uFE0F\u20E3",
        "\u0036\uFE0F\u20E3",
        "\u0037\uFE0F\u20E3",
        "\u0038\uFE0F\u20E3",
        "\u0039\uFE0F\u20E3",
        "\U0001F1E6",
        "\U0001F1E7",
        "\U0001F1E8",
        "\U0001F1E9",
        "\U0001F1EA",
        "\U0001F1EB",
        "\U0001F1EC",
        "\U0001F1ED",
        "\U0001F1EE",
        "\U0001F1EF",
        "\U0001F1F0",
    ]

    channel_id = str(ctx.channel.id)

    if channel_id not in conversations:
        conversations[channel_id] = []

    # Update conversation history with the user's question
    conversations[channel_id].append({"role": "user", "content": question})

    # Get the response from OpenAI API with conversation history
    openai_response = get_openai_response(conversations[channel_id], question)

    conversations[channel_id].append({"role": "system", "content": openai_response})

    # Process the movie list
    movie_titles_map = check_for_movie_title_in_string(openai_response)

    # Initialize an empty list to hold the new response
    new_response_lines = []

    global_emoji_index = (
        0  # Global index to keep track of emoji assignment across all segments
    )

    # Split the OpenAI response into lines
    openai_response_lines = openai_response.split("\n")

    # Process each line
    for line in openai_response_lines:
        matches = re.findall(r"\*([^*]+)\* \((\d{4})\)", line)
        if matches:
            # Reset the line text to rebuild it with emojis
            new_line = line

            for match in matches:
                original_title, year = match
                tmdb_title = movie_titles_map.get(original_title, original_title)

                # Ensure we don't run out of emojis
                if global_emoji_index < len(emojis):
                    emoji = emojis[global_emoji_index]
                    global_emoji_index += 1
                else:
                    emoji = ""  # Default to no emoji if we run out

                # Replace the movie title in the line with the title and emoji
                movie_placeholder = f"*{original_title}* ({year})"
                new_line = new_line.replace(
                    movie_placeholder, f"{emoji} {movie_placeholder}", 1
                )

            new_response_lines.append(new_line)
        else:
            new_response_lines.append(line)

    # Join the new lines to form the updated response
    response_chunk = "\n".join(new_response_lines)

    # Custom function to split text at spaces
    def split_text(text, max_length):
        segments = []
        while len(text) > max_length:
            split_index = text.rfind(
                " ", 0, max_length
            )  # Find the last space within the limit
            if split_index == -1:  # No space found, use the max_length
                split_index = max_length
            segments.append(text[:split_index])  # Split at the space
            text = text[
                split_index:
            ].lstrip()  # Remove leading spaces from the next part
        segments.append(text)  # Add the remaining part of the text
        return segments

    # Splitting response into chunks at spaces
    response_segments = split_text(response_chunk, 1900)

    last_msg = None
    total_movie_count = 0

    for response_segment in response_segments:
        msg = await ctx.send(f"\n{response_segment}")
        last_msg = msg  # Update the last message
        total_movie_count += (
            response_segment.count("*") // 2
        )  # Count the pairs of asterisks

    # Add reactions only to the last message
    if last_msg:
        for i in range(min(total_movie_count, len(emojis))):
            await last_msg.add_reaction(emojis[i])
        segment_emoji_map[last_msg.id] = emojis[
            :total_movie_count
        ]  # Store only the used emojis for the last segment

    message_movie_map[last_msg.id] = movie_titles_map


@client1.event
async def on_reaction_add(reaction, user):
    # Ignore reactions added by the bot itself
    if user == client1.user:
        return

    # Check if the reaction is on a message that contains movies
    if reaction.message.id in message_movie_map:
        # Check if the message has associated emojis in the map
        if reaction.message.id in segment_emoji_map:
            # Retrieve the emojis associated with the message
            message_emojis = segment_emoji_map[reaction.message.id]
            # Find the index of the emoji in the message's emojis
            emoji_index = (
                message_emojis.index(str(reaction.emoji))
                if str(reaction.emoji) in message_emojis
                else -1
            )

            # Debugging log to check the emoji index and the list of movies
            print(
                f"Emoji Index: {emoji_index}, Movies List: {message_movie_map[reaction.message.id]}"
            )

            # Validate the emoji index and check it's within the range of the movies list
            if emoji_index != -1 and emoji_index < len(
                message_movie_map[reaction.message.id].values()
            ):
                # Retrieve the selected movie based on the emoji index
                selected_movie = list(message_movie_map[reaction.message.id].values())[
                    emoji_index
                ]

                # Additional debugging log to check the selected movie
                print(f"Selected Movie: {selected_movie}")

                # Proceed with adding the movie to Radarr if it's a valid selection
                if selected_movie:
                    try:
                        # Search for the movie in Radarr
                        search = radarr.search_movies(selected_movie)
                        # Add the movie to Radarr if found
                        if search:
                            search[0].add("/data/media/movies", RADARR_QUALITY)
                            await reaction.message.channel.send(
                                f"'{selected_movie}' has been added to Radarr."
                            )
                        else:
                            # Handle the case where the movie is not found in Radarr
                            await reaction.message.channel.send(
                                f"'{selected_movie}' not found in Radarr."
                            )
                    except arrapi.exceptions.Exists:
                        # Handle if the movie already exists in Radarr
                        await reaction.message.channel.send(
                            f"You already have '{selected_movie}' in Radarr."
                        )
                    except Exception as e:
                        # Handle other exceptions
                        await reaction.message.channel.send(f"Error: {e}")
            else:
                # Log if the emoji index is out of range or invalid
                print("Emoji index out of range or invalid.")


def start_discord_bot_up():
    client1.run(DISCORD_TOKEN)


if __name__ == "__main__":
    print("Running that t hing in")
    # Load configuration
    config = load_config()

    # Check if the Discord bot should be started
    config = load_or_create_config()
    # Check if the start_discord_bot_on_launch key is true
    if config.get("start_discord_bot_on_launch", True):
        required_keys = [
            "tmdb_api_key",
            "radarr_url",
            "radarr_api_key",
            "openai_api_key",
            "discord_token",
            "radarr_quality",
            "selected_model",
            "max_chars",
            "discord_channel",
        ]
        if all(key in config and config[key] for key in required_keys):
            client1.run(DISCORD_TOKEN)
    else:
        print("Discord bot startup is disabled in the configuration.")
