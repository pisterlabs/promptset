import asyncio
import os
import nextcord
from nextcord.ext import commands
import openai
from pymongo import MongoClient
import urllib.parse

openai.api_key = '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'

# not working at the moment


# MongoDB connection details
username = urllib.parse.quote_plus("apwade75009")
password = urllib.parse.quote_plus("Celina@12")
cluster = MongoClient(f"mongodb+srv://{username}:{password}@quantaai.irlbjcw.mongodb.net/")
db = cluster["QuantaAI"]  # Replace "YourNewDatabaseName" with your desired database name
banhammer_collection = db["banhammer"]

class BanHammer(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    async def is_ban_hammer_enabled(self, guild_id):
        result = banhammer_collection.find_one({"_id": guild_id})
        if result:
            return result.get("enabled", False)
        return False

    async def set_ban_hammer_enabled(self, guild_id, enabled):
        banhammer_collection.update_one(
            {"_id": guild_id},
            {"$set": {"enabled": enabled}},
            upsert=True
        )

    async def warn_user(self, user, guild):
        user_id = str(user.id)
        result = banhammer_collection.find_one({"_id": user_id})
        warns = result.get("warns", 0) if result else 0
        warns += 1

        if warns >= 5:
            # Temporarily ban the user for 3 days
            await user.ban(reason="Inappropriate messages detected by Ban Hammer", delete_message_days=0)
            await asyncio.sleep(3 * 24 * 60 * 60)  # Wait for 3 days
            await guild.unban(user)
            warns = 0

        banhammer_collection.update_one(
            {"_id": user_id},
            {"$set": {"warns": warns}},
            upsert=True
        )

    @nextcord.slash_command(name="banhammer", description="Enable or disable the Ban Hammer")
    @commands.has_permissions(administrator=True)
    async def banhammer(self, interaction: nextcord.Interaction):
        guild_id = str(interaction.guild.id)
        enabled = await self.is_ban_hammer_enabled(guild_id)

        if enabled:
            await self.set_ban_hammer_enabled(guild_id, False)
            message = "Ban Hammer has been disabled."
        else:
            await self.set_ban_hammer_enabled(guild_id, True)
            message = "Ban Hammer has been enabled."

        await interaction.send("Processing...")

        # Define the animation frames using ASCII art
        animation_frames = [
            "⚡️",
            "⚡️⚡️⚡️",
            "⚡️⚡️⚡️⚡️⚡️",
            "⚡️⚡️⚡️⚡️⚡️⚡️⚡️",
            "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️",
            "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️",
            "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️",
            "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️",
            "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️",
            "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️"
        ]

        # Send the animation frames with a delay between them
        for frame in animation_frames:
            await asyncio.sleep(0.5)  # Adjust the delay between frames if needed
            await interaction.followup.send(frame)

        await interaction.followup.send(message)


    @commands.Cog.listener()
    async def on_message(self, message):
        print(f"Received message: {message.content}")
        if message.author == self.bot.user:
            return

        if not await self.is_ban_hammer_enabled(message.guild.id):
            return

        try:
            response = openai.Moderation.create(input=message.content)
            flagged = response['results'][0]['flagged']
        except Exception as e:
            print(f"Error while calling OpenAI API: {e}")
            return

        if flagged:
            await self.warn_user(message.author, message.guild)
            result = banhammer_collection.find_one({"_id": str(message.author.id)})
            warns = result.get("warns", 0) if result else 0

            if warns == 5:
                await message.channel.send(f"{message.author.name} has been temporarily banned for 3 days due to inappropriate messages.")
            else:
                warning_message = f"{message.author.name}, this is a warning. You have sent an inappropriate message. "\
                                f"You have {warns} out of 5 warnings. "\
                                f"Reaching 5 warnings will result in a temporary ban for 3 days."
                await message.channel.send(warning_message)
                try:
                    await message.author.send(warning_message)
                except nextcord.errors.Forbidden:
                    pass



def setup(bot):
    bot.add_cog(BanHammer(bot))
    print("BanHammer Ready!")
