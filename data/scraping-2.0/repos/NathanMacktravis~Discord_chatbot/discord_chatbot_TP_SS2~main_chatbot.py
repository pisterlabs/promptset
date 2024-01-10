import discord
from discord import Intents
from weather_chatbot import main_chatbot_weather
from GPT_bot import openai_chatbot
from movie_chatbot import *

# Token of the weather wit model
WIT_WEATHER_KEY = "WIT_WEATHER_TOKEN"

# Token of the API
TMDB_api_key = "TMDB_TOKEN"

# token of the movie wit model
WIT_MOVIE_KEY = "WIT_MOVIE_TOKEN" #(new)

# Discord token
discord_key = "DISCORD_TOKEN"

intents = Intents.default()
intents.message_content = True

# Initializing Wit.ai clients
wit_movie_client = Wit(WIT_MOVIE_KEY)
wit_weather_client = Wit(WIT_WEATHER_KEY)


# Function to detect the intention (movie, weather, or unknown)
def detect_intent(user_message):
    movie_response = wit_movie_client.message(user_message)
    weather_response = wit_weather_client.message(user_message)

    # Comparison of confidences to determine intent
    movie_confidence = movie_response['intents'][0]['confidence'] if movie_response['intents'] else 0
    weather_confidence = weather_response['intents'][0]['confidence'] if weather_response['intents'] else 0

    if movie_confidence > weather_confidence:
        return 'movie'
    elif weather_confidence > movie_confidence:
        return 'weather'
    else:
        return 'unknown'


class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged in as ' + str(self.user.name))
        print('Chatbot Id :: ' + str(self.user.id))

    async def on_message(self, message):
        # Check if the message is from a bot to avoid infinite loops
        if message.author.bot:
            return

        # Process the content of the user's message
        user_message = message.content
        user_name = str(message.author).split('#')[0]
        print(f"{user_name} : {user_message}")

        # Detect intent (movie, weather, or unknown)
        intent = detect_intent(user_message)
        response_message = ""  # Initialisation avec une chaîne vide
        
        if intent == 'movie':
            # Extract the names of the movies in the user message
            movie_names = extract_movie_names(user_message)

            # Get movie information for each movie
            if movie_names:
                responses_movies, posters = movie_infos(movie_names)
                for response_movie, poster in zip(responses_movies, posters):
                    if poster:
                        await message.channel.send(file=discord.File(poster, 'movie_poster.png'))
                    await message.channel.send(response_movie)
                print("Here we have a movie question")

        elif intent == 'weather':
            # Get weather information for the message
            response_message = f"Hi {message.author.mention}, {main_chatbot_weather(user_message)}"
            print("Here we have a weather question")

        else:
            # If the intent is unknown, use ChatGPT from OpenAI
            response_message = f"Hi {message.author.mention}, {await openai_chatbot(user_message)}"
            print("Here we have an unknown question")

        await message.channel.send(response_message)



# Initialise le client Discord
discord_client = MyClient(intents=intents)
discord_client.run(discord_key)
