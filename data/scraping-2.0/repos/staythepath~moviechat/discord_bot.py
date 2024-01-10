import discord
from discord.ext import commands
import openai
from tmdbv3api import TMDb, Movie
import re
import asyncio
from arrapi import RadarrAPI
import arrapi.exceptions
import yaml
import os
import time
from openai_chat_manager import OpenAIChatManager
from config_manager import ConfigManager


class DiscordBot(commands.Bot):
    def __init__(self, config_path="config.yaml"):
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        super().__init__(command_prefix="!!", intents=intents)

        config_manager = ConfigManager(config_path)
        self.config = config_manager.load_config()
        self.message_movie_map = {}
        self.segment_emoji_map = {}
        self.conversations = {}
        self.tmdb = TMDb()
        self.tmdb.api_key = self.config["tmdb_api_key"]
        self.movie = Movie()
        self.radarr = RadarrAPI(
            self.config["radarr_url"], self.config["radarr_api_key"]
        )
        self.openai_client = openai.Client(api_key=self.config["openai_api_key"])

        self.add_listener(self.on_ready)
        self.add_listener(self.on_reaction_add)

    async def on_ready(self):
        print(f"Logged in as {self.user.name}")
        discord_channel_id = self.config.get("discord_channel_id")

        if discord_channel_id:
            channel = self.get_channel(int(discord_channel_id))
            if channel:
                await channel.send("I'm up and running!")
            else:
                print(f"Channel with ID {discord_channel_id} not found.")
        else:
            print("Discord channel ID not set in config.")

    @commands.command(name="ask")
    async def ask(self, ctx, *, question):
        print("ask function that sucks is running now........")
        conversations = {}
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

        print("Starting the open ai stuff....??")
        # Update conversation history with the user's question
        conversations[channel_id].append({"role": "user", "content": question})

        # Get the response from OpenAI API with conversation history
        openai_response = OpenAIChatManager.get_openai_response(
            conversations[channel_id], question
        )

        conversations[channel_id].append({"role": "system", "content": openai_response})

        # Process the movie list
        movie_titles_map = OpenAI.check_for_movie_title_in_string(openai_response)

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
            self.segment_emoji_map[last_msg.id] = emojis[
                :total_movie_count
            ]  # Store only the used emojis for the last segment

        self.message_movie_map[last_msg.id] = movie_titles_map

    async def on_reaction_add(self, reaction, user):
        # Ignore reactions added by the bot itself
        if user == client1.user:
            return

        # Check if the reaction is on a message that contains movies
        if reaction.message.id in self.message_movie_map:
            # Check if the message has associated emojis in the map
            if reaction.message.id in self.segment_emoji_map:
                # Retrieve the emojis associated with the message
                message_emojis = self.segment_emoji_map[reaction.message.id]
                # Find the index of the emoji in the message's emojis
                emoji_index = (
                    message_emojis.index(str(reaction.emoji))
                    if str(reaction.emoji) in message_emojis
                    else -1
                )

                # Debugging log to check the emoji index and the list of movies
                print(
                    f"Emoji Index: {emoji_index}, Movies List: {self.message_movie_map[reaction.message.id]}"
                )

                # Validate the emoji index and check it's within the range of the movies list
                if emoji_index != -1 and emoji_index < len(
                    self.message_movie_map[reaction.message.id].values()
                ):
                    # Retrieve the selected movie based on the emoji index
                    selected_movie = list(
                        self.message_movie_map[reaction.message.id].values()
                    )[emoji_index]

                    # Additional debugging log to check the selected movie
                    print(f"Selected Movie: {selected_movie}")

                    # Proceed with adding the movie to Radarr if it's a valid selection
                    if selected_movie:
                        try:
                            # Search for the movie in Radarr
                            search = self.radarr.search_movies(selected_movie)
                            # Add the movie to Radarr if found
                            if search:
                                search[0].add(
                                    "/data/media/movies", self.config["tmdb_api_key"]
                                )
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

    def start_bot(self):
        # Retrieve the Discord token from the configuration
        discord_token = self.config.get("discord_token")
        print(f"Starting bot with token: {discord_token}")  # Debugging print statement
        if discord_token:
            # Start the bot with the provided token
            self.run(discord_token)
        else:
            print("Discord token not found in config.")

    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.CommandNotFound):
            await ctx.send(f"Command not found: {ctx.message.content}")
        else:
            await ctx.send(f"An error occurred: {error}")


if __name__ == "__main__":
    bot = DiscordBot()
    print("Creating DiscordBot instance...")
    bot.start_bot()
