import asyncio
import os
import urllib.parse

import nextcord
from nextcord.ext import commands
from langdetect import detect
from pymongo import MongoClient
import openai

# MongoDB connection details
username = urllib.parse.quote_plus("apwade75009")
password = urllib.parse.quote_plus("Celina@12")
cluster = MongoClient(f"mongodb+srv://{username}:{password}@quantaai.irlbjcw.mongodb.net/")
db = cluster["QuantaAI"]

class TranslationCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.enabled = True
        self.chat_model = None
        self.translation_settings_collection = db["translation_settings"]

    def load_settings(self):
        result = self.translation_settings_collection.find_one()
        return result if result else {}

    def save_settings(self, settings):
        self.translation_settings_collection.replace_one({}, settings, upsert=True)

    async def translate_message(self, message):
        openai.api_key = os.getenv("OPENAI_API_KEY")

        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Translate the following text to English: {message}",
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.5,
            )
        except Exception as e:
            print(f"Error: {e}")
            return "Translation error"

        translated_message = response.choices[0].text.strip()
        return translated_message

    def is_english(self, text):
        try:
            lang = detect(text)
            return lang == 'en'
        except:
            return False

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            if message.content.startswith("Translation is now"):
                return
            else:
                return

        settings = self.load_settings()
        guild_id = str(message.guild.id)
        if guild_id not in settings or not settings[guild_id]["enabled"]:
            return

        # Check if the translation channel is set and if the message is in the correct channel
        translation_channel_id = settings[guild_id].get("translation_channel")
        if translation_channel_id and message.channel.id != translation_channel_id:
            return

        if message.content and not self.is_english(message.content):
            translated_message = await self.translate_message(message.content)
            if translated_message.lower() != message.content.lower():
                await message.channel.send(f"{message.author.mention} said (translated): {translated_message}")

    @nextcord.slash_command(name="servertranslation", description="Enable or disable server translation")
    async def main(self, interaction: nextcord.Interaction):
        pass

    @main.subcommand(name="toggler", description="Enable or disable server translation")
    @commands.has_permissions(administrator=True)
    async def toggle_translation(self, interaction: nextcord.Interaction):
        await interaction.response.defer()
        settings = self.load_settings()
        guild_id = str(interaction.guild.id)

        if guild_id not in settings:
            settings[guild_id] = {"enabled": False}

        settings[guild_id]["enabled"] = not settings[guild_id]["enabled"]
        self.save_settings(settings)

        await interaction.send(f"Translation is now {'enabled' if settings[guild_id]['enabled'] else 'disabled'}")

        if not settings[guild_id]["enabled"]:
            await asyncio.sleep(1)
            settings[guild_id]["enabled"] = False
            self.save_settings(settings)

    @main.subcommand(name="setchannel", description="Set the channel for server translation")
    @commands.has_permissions(administrator=True)
    async def set_translation_channel(self, interaction: nextcord.Interaction, channel: nextcord.TextChannel):
        settings = self.load_settings()
        guild_id = str(interaction.guild.id)

        if guild_id not in settings:
            settings[guild_id] = {"enabled": False, "translation_channel": None}

        settings[guild_id]["translation_channel"] = channel.id
        self.save_settings(settings)

        await interaction.send(f"Translation channel has been set to {channel.mention}")

    @main.subcommand(name="setmodel", description="Set the model for server translation")
    @commands.has_permissions(administrator=True)
    async def set_chat_model(self, interaction: nextcord.Interaction, model_key):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.chat_model = openai.Completion.create(engine=model_key)

    @main.subcommand(name="disabler", description="Disable server translation")
    @commands.has_permissions(administrator=True)
    async def disable(self, interaction: nextcord.Interaction):
        self.enabled = False

    @main.subcommand(name="enabler", description="Enable server translation")
    @commands.has_permissions(administrator=True)
    async def enable(self, interaction: nextcord.Interaction):
        self.enabled = True

def setup(bot):
    bot.add_cog(TranslationCog(bot))
