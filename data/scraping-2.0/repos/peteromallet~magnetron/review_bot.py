from dotenv import load_dotenv
import os 
import json
import pandas as pd
import discord 
import uuid
import json
import asyncio
from discord.ext import commands
import openai
import logging

# Set up logging
logging.basicConfig(filename='bot.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

logging.info('This is an info message')
logging.error('This is an error message')

load_dotenv(override=True)  # This loads the environment variables from .env file

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

class DiscordDataBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.input_df = pd.read_csv("output.csv")       
        self.user_history = {}

    def get_random_row(self):
        to_be_reviewed_df = self.input_df[self.input_df['status'] == 'to_be_reviewed']        
        if to_be_reviewed_df.empty:
            return None        
        random_row = to_be_reviewed_df.sample()
        random_row_json = json.loads(random_row.to_json(orient="records"))[0]
        return random_row_json
    
    async def save_to_csv(self, user_id, uuid, rating, feedback=""):                
        # Refresh the DataFrame from the source CSV file
        self.input_df = pd.read_csv("output.csv")

        new_entry = {"user_id": user_id, "uuid": uuid, "rating": rating}
        filtered_df = self.input_df[self.input_df['uuid'] == uuid]
        if not filtered_df.empty:
            row_index = filtered_df.index[0]
            self.input_df.at[row_index, 'user_rating'] = json.dumps([new_entry])
            self.input_df.at[row_index, 'status'] = 'reviewed'
            self.input_df.at[row_index, 'user_fix_suggestions'] = feedback
            self.input_df.to_csv("output.csv", index=False)
        else:
            raise ValueError(f"No row with uuid {uuid} found in input_df")

class FeedbackView(discord.ui.View):
    def __init__(self, ctx, id, original_uuid, current_label = "", original_message_id = 0):
        super().__init__(timeout=1200)
        self.ctx = ctx
        self.id = id
        self.original_uuid = original_uuid
        self.current_label = current_label
        self.original_message_id = original_message_id

        # Check states and update the styles accordingly
        rating = self.ctx.bot.user_history[self.ctx.author.id][self.id]["rating"] if self.ctx.author.id in self.ctx.bot.user_history else 0
        both_good_style = discord.ButtonStyle.success if rating == 2 else discord.ButtonStyle.secondary
        image_good_prompt_bad_style = discord.ButtonStyle.success if rating == 1 else discord.ButtonStyle.secondary
        both_bad_style = discord.ButtonStyle.danger if rating == 0 else discord.ButtonStyle.secondary

        self.both_bad_button = discord.ui.Button(label="Not good enough", emoji="❌", style=both_bad_style, row=0)
        self.both_bad_button.callback = self.both_bad
        self.add_item(self.both_bad_button)

        self.image_good_prompt_bad_button = discord.ui.Button(label="I CAN FIX IT!!!", emoji="😭", style=image_good_prompt_bad_style, row=0)
        self.image_good_prompt_bad_button.callback = self.image_good_prompt_bad
        self.add_item(self.image_good_prompt_bad_button)

        self.both_good_button = discord.ui.Button(label="Images + prompt good", emoji="✅", style=both_good_style, row=0)
        self.both_good_button.callback = self.both_good
        self.add_item(self.both_good_button)

    async def skip(self, interaction: discord.Interaction):
        await interaction.response.send_message("Skipping image pair, sending next one...")
        await send_image_pair(self.ctx) 

    async def both_good(self, interaction: discord.Interaction):    
        await interaction.response.defer()

        self.both_good_button.style = discord.ButtonStyle.blurple
        self.image_good_prompt_bad_button.style = discord.ButtonStyle.secondary
        self.both_bad_button.style = discord.ButtonStyle.secondary
        await interaction.message.edit(view=self)
        bot.user_history[self.ctx.author.id][self.id]["rating"] = 2
        await bot.save_to_csv(self.ctx.author.id, self.original_uuid, 2)
        await interaction.message.delete()
        await send_image_pair(self.ctx)

    async def image_good_prompt_bad(self, interaction: discord.Interaction):    
        await interaction.response.defer()

        self.image_good_prompt_bad_button.style = discord.ButtonStyle.blurple
        self.both_good_button.style = discord.ButtonStyle.secondary
        self.both_bad_button.style = discord.ButtonStyle.secondary
        await interaction.message.edit(view=self)
                
        await interaction.channel.send("How would you change it? Just specify changes briefly, GPT will do the rest.")
        
        def check(message):
            return message.author == interaction.user and message.channel == interaction.channel

        try:
            # Wait for the user's reply
            user_reply = await bot.wait_for('message', timeout=30.0, check=check)
        except asyncio.TimeoutError:
            await interaction.channel.send("You took too long to reply!")        

        bot.user_history[self.ctx.author.id][self.id]["rating"] = 1
        await bot.save_to_csv(self.ctx.author.id, self.original_uuid, 1, user_reply.content)
        await interaction.message.delete()
        await send_image_pair(self.ctx)


    async def both_bad(self, interaction: discord.Interaction):
        await interaction.response.defer()

        self.both_bad_button.style = discord.ButtonStyle.blurple
        self.image_good_prompt_bad_button.style = discord.ButtonStyle.secondary
        self.both_good_button.style = discord.ButtonStyle.secondary
        await interaction.message.edit(view=self)

        bot.user_history[self.ctx.author.id][self.id]["rating"] = 0
        await bot.save_to_csv(self.ctx.author.id, self.original_uuid, 0)
        await interaction.message.delete()
        await send_image_pair(self.ctx)

    async def report(self, interaction: discord.Interaction):
        await interaction.response.send_message("Image pair reported")
        await send_image_pair(self.ctx)

if __name__ == '__main__':
    intents = discord.Intents.default()
    intents.message_content = True 
    bot = DiscordDataBot(command_prefix="!",intents=intents)

    @bot.event
    async def on_ready():
        logging.info(f"We have logged in as {bot.user}")
        print(f"We have logged in as {bot.user}")

    @bot.command(description="Get Image Pair")        
    async def start_running(ctx):
        logging.info(f"start_running command called by {ctx.author}")
        await send_image_pair(ctx)

    async def send_image_pair(ctx):
        id = f"{ctx.author.id}_{uuid.uuid4()}"
        image_pair = bot.get_random_row()
        
        if image_pair is None:
            await asyncio.sleep(300)  # Wait for 5 minutes
            return
        
        if ctx.author.id not in bot.user_history:
            bot.user_history[ctx.author.id] = {}

        bot.user_history[ctx.author.id][id] = {
            "user_id" : ctx.author.id,
            "uuid": image_pair["uuid"],
            "image_0_location": image_pair["image_0_location"],            
            "image_1_location": image_pair["image_1_location"],            
            "caption": image_pair["caption"], 
            "rating": -1, 
        }
        
        try:
            caption_list = json.loads(image_pair["caption"])
        except json.decoder.JSONDecodeError:
            print("Error: Invalid JSON in caption")
            caption_list = []

        # Initialize an empty string to store the formatted caption
        formatted_caption = ""

        # Iterate over the list of dictionaries
        for item in caption_list:
            try:
                # Each dictionary has only one key-value pair, so we can get them like this
                for key, value in item.items():
                    # Add the key and value to the formatted caption, with the key in bold
                    formatted_caption += f"**{key}**: {value}\n"
            except AttributeError:
                
                continue  # Skip this item and move on to the next one

        # Now you can use the formatted caption in your message
        await ctx.send(
            content=f'Please vote on if this is a valid description of the motion between the top and bottom image or not: \n\n{image_pair["image_0_location"]} {image_pair["image_1_location"]} \n\nInstructions:\n\n{formatted_caption}\n-',
            view=FeedbackView(ctx, id=id, original_uuid=image_pair["uuid"])  # pass the original uuid
        )

    print("Running...")
    bot.run(DISCORD_TOKEN)