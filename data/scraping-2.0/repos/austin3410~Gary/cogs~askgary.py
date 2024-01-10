
import discord
from discord.commands import slash_command  # Importing the decorator that makes slash commands.
from discord.ext import commands
import openai
from openai import OpenAI
from discord.commands import Option
import io
import aiohttp
import os
import pickle

# This function pulls a file from memory so we can send it to Discord without saving it.
async def get_image_file(img_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(img_url) as resp:
            if resp.status != 200:
                return False
            
            data = io.BytesIO(await resp.read())

            return data

# This function creates a user profile for OpenAI usage.
async def create_openai_user(ctx, openai_api_token, openai_org):
    try:
        with open(f"files//OPENAI_USERS//{ctx.author.id}.pickle", "wb") as file:
            openai_user_file = {"openai_api_token": openai_api_token, "openai_org": openai_org}
            pickle.dump(openai_user_file, file)
        
        return "Your OpenAI token and Orginization have been saved successfully! You should now be able to use /image!"
    except Exception as e:
        return f"Something went wrong, we were unable to save your user file:\n{e}"

# This function fetches a user profile for OpenAI usage.
async def get_openai_user(ctx):
    for root, folders, files in os.walk("files//OPENAI_USERS"):
        for file in files:
            if str(ctx.author.id) in file:
                with open(f"{root}//{file}", "rb") as file:
                    openai_user_file = pickle.load(file)
                    return openai_user_file
    
    return None

# This is the class that controls how the /image previw window looks and works.
class ImageViewer(discord.ui.View):
    def __init__(self, timeout=None, bot=None, ctx=None, prompt=None, size=None):
        super().__init__(timeout=timeout)
        self.bot = bot
        self.ctx = ctx
        self.prompt = prompt
        self.size = size

    # This button simply sends the image to the person who clicked the button.
    @discord.ui.button(label="Save", emoji="💾", style=discord.ButtonStyle.primary)
    async def save_image(self, button, interaction):
        msg = await interaction.user.send(interaction.message.embeds[0].image.url)
        return await interaction.response.send_message(f"I've DM'd you this image!\n{msg.jump_url}", ephemeral=True)
    
    # This button adds the generated image to a <Gallery> channel. See config-example.py for an example.
    @discord.ui.button(label="Send to Gallery", emoji="⭐", style=discord.ButtonStyle.success, custom_id="gallery_button")
    async def send_to_gallery(self, button, interaction):
        original_embed = interaction.message.embeds[0]
        embed = discord.Embed(title=original_embed.title, color=0xffd53e)
        embed.set_image(url=original_embed.image.url)
        embed.author.name = f"Submitted by {interaction.user.name}"
        embed.add_field(name="Prompt", value=self.prompt, inline=True)
        embed.add_field(name="Size", value=self.size, inline=True)
        
        msg = await self.bot.gallery_channel.send(embed=embed)
        
        await interaction.message.add_reaction(emoji="⭐")
        self.children[1].disabled = True
        self.children[1].label = "Already Sent to Gallery"
        
        await interaction.message.edit(view=self)
        return await interaction.response.send_message(f"<@{interaction.user.id}> has submitted this image to {msg.jump_url}")

    """# This button is a TEST button which prints a bunch of debug info on the embeded image.
    @discord.ui.button(label="TEST", style=discord.ButtonStyle.primary)
    async def TEST(self, button: discord.Button, interaction: discord.Interaction):
        print(interaction.message)
        print(interaction.message.embeds[0])
        print(interaction.message.embeds[0].image)
        print(interaction.message.embeds[0].image.url)"""

# This class handles talking to Gary.
class AskGary(commands.Cog):
    # Inits the bot instance so we can do things like send messages and get other Discord information.
    def __init__(self, bot):
        self.bot = bot
        self.garys_openai_client = OpenAI(api_key=self.bot.openai_key, organization=self.bot.openai_orginization)
        

        # Adjust Gary's personality to be more or less true to the show.
        self.bot_personality = "You are Gary the snail from the TV Show SpongeBob Squarepants in a Discord server called Bikini Bottom. You're very helpful and enjoy answering everyones questions."

    # We need this function incase the response_text is longer than Discords tiny 2000/message character limit.    
    def split_response_text(self, response_text):
        
        # First we check if the response is actually over the limit.
        # If it's not, we'll just pass it as is.
        if len(response_text) > 2000:
            
            # This basically lets us remember how much of the message we've split and how much needs splitting.
            cursor_location = 0
            
            # This stores our message "chunks".
            messages = []
            
            # This runs through the response and creates smaller messages, 2000 characters at a time.
            while cursor_location < len(response_text):
                
                message_block = response_text[cursor_location:(cursor_location + 2000)]
                messages.append(message_block)
                cursor_location += 2000
            
            return messages
        else:
            return response_text

    
    # This function handles generating message blocks that get sent to ChatGPT and formating the questions/responses.
    async def generate_response(self, ctx, history=None):

        # This removes Gary's bot ID before sending it to ChatGPT.
        new_message = str(ctx.content).replace(f"<@{self.bot.id}> ", "")

        # This is the start of the message block that will inform ChatGPT of how the conversation is going so far.
        messages = [{"role": "system", "content": self.bot_personality}]

        # If history is present that means we're in an ongoing conversation.
        if history != None:

            # This is useful to remind ChatGPT of who started the conversation.
            history[0]["content"] = f"<@{history[0]['author_id']}> asks, hey Gary," + history[0]["content"]
            
            # This loops through all of the messages in the history and recreates the conversation from the beginning.
            for msg in history:

                # IF the author is Gary.
                if str(msg["author_id"]) == str(self.bot.id):

                    # We filter out Gary's automatic, "Meow." messages.
                    if msg["content"] == "Meow.":
                        pass
                    
                    # We include anything he says in the past as himself.
                    else:
                        messages.append({"role": "assistant", "content": msg["content"]})
                
                # Anything else must've come from a human (or possibly another bot.)
                else:
                    messages.append({"role": "user", "content": msg["content"], "name": str(msg['author_id'])})
            
            # Once all of the past messages are re-built, we append our new message to the message string.
            messages.append({"role": "user", "content": new_message, "name": str(ctx.author.id)})
        else:

            # If there is no history that means this is a new conversation and we need to make sure the message is formatted like a question for good results.
            messages.append({"role": "user", "content": f"<@{ctx.author.id}> asks, hey Gary," + new_message, "name": str(ctx.author.id)})

        # This is the actual request for a chat completion.
        try:
            response = self.garys_openai_client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=messages, # This is our contructed block of past messages + our new message.
                temperature=0.9,
                max_tokens=15000,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
        except Exception as e:
            print(str(e))
            if "safety system" in str(e):
                return "Your prompt was rejected by the safety team. Try altering your request!"
            elif "token" in str(e):
                return "I've reached my token limit. Please start a new conversation."
            elif str(e).startswith("Error code: 400"):
                return "I've reached my billing limit for the month. If you would like to help raise the billing limit, visit the Server Shop!"
            else:
                return "I couldn't complete the request at this time, please try again!"
        
        # The actual response text.
        response_text = str(response.choices[0].message.content)
        
        # This made user ids parsable by Discord but seems to be no longer needed.
        #response_text = str(response_text).replace(str(ctx.author.id), f"<@{ctx.author.id}>")

        return response_text
    
    # The on_message hook that captures new messages and determines if they're ai chat messages.
    @commands.Cog.listener()
    async def on_message(self, ctx):

        # First and foremost, if the message is from Gary, ignore it.
        if str(ctx.author.id) != str(self.bot.id):

            # Now if the message is from a regular text channel
            if ctx.channel.type in [discord.ChannelType.text]:
                # If the message starts by @ing Gary.
                if ctx.content.startswith(f"<@{self.bot.id}>"):

                    # This is the start of a new conversation thread.
                    # Generate response text, create a thread, send the response.
                    response_text = await self.generate_response(ctx)
                    response_text = self.split_response_text(response_text)
                    thread_name = f"{ctx.author.name}'s Thread"
                    convo_thread = await ctx.create_thread(name=thread_name)
                    if isinstance(response_text, list):
                        for message in response_text:
                            await convo_thread.send(message)
                    else:
                        return await convo_thread.send(response_text)

            # If the channel is a thread, voice_text, or private/dm channel.
            if ctx.channel.type in [discord.ChannelType.public_thread, discord.ChannelType.voice, discord.ChannelType.private]:
                # If there's a reference that means the message is a reply to an earlier message, which makes this an on-going conversation.
                if ctx.reference:

                    # Since this is an on-going convo, we need to capture and rebuild the entire conversation up to this point.
                    history = []
                    target_message_id = ctx.reference.message_id
                    targeted_message = await ctx.channel.fetch_message(target_message_id)

                    # This makes sure the replied to message was a message we sent. If it's not, we'd be replying to someone elses thread... rude.
                    if str(targeted_message.author.id) != str(self.bot.id):
                        return

                    # Start typing...
                    async with ctx.channel.typing():
                        # This loop recursively bounces up the message chain to the first message and captures required data along the way.
                        while True:
                            next_hop_msg = await ctx.channel.fetch_message(target_message_id)

                            # If the next message == None, that means we're at the start of the message chain.
                            if next_hop_msg == None:
                                break
                            
                            # This captures the required info.
                            history.insert(0, {"author_name": next_hop_msg.author.name, "author_id": next_hop_msg.author.id, "content": next_hop_msg.content})

                            # This tries to queue up the next message in the chain. If it breaks that means there are no more messages and we are done.
                            try:
                                target_message_id = next_hop_msg.reference.message_id
                            except:
                                break
                            
                        # Now that we have our convo history, we can generate a response from ChatGPT.
                        response_text = await self.generate_response(ctx, history=history)
                        response_text = self.split_response_text(response_text)

                        # This replies instead of creating a thread since we're already in a thread, or the channel doesn't support a thread.
                        if isinstance(response_text, list):
                            for message in response_text:
                                await ctx.reply(message)
                        else:
                            return await ctx.reply(response_text)

                # If there's no reference, that means this is the start of a new conversation.
                # Depending on where this conversation is taking place we either NEED to be @ mentioned... or not.
                else:
                    # If Gary is @ mentioned and the channel isn't a DM, we're in a voice-text channel.
                    if ctx.content.startswith(f"<@{self.bot.id}>") and ctx.channel.type != discord.ChannelType.private:
                        # Start typing...
                        async with ctx.channel.typing():
                            response_text = await self.generate_response(ctx)
                            response_text = self.split_response_text(response_text)
                            if isinstance(response_text, list):
                                for message in response_text:
                                    await ctx.reply(message)
                            else:
                                return await ctx.reply(response_text)
                    
                    # If Gary isn't @ mentioned and it is a DM, then that's fine, we can proceed as normal. Plus we know it's not a on-going convo because it's NOT a reply.
                    elif ctx.channel.type == discord.ChannelType.private:
                        # Start typing...
                        async with ctx.channel.typing():
                            response_text = await self.generate_response(ctx)
                            response_text = self.split_response_text(response_text)
                            if isinstance(response_text, list):
                                for message in response_text:
                                    await ctx.reply(message)
                            else:
                                return await ctx.reply(response_text)
    
    async def is_dm(ctx):
        if ctx.channel.type == discord.ChannelType.private:
            return True
        else:
            await ctx.author.send("Please use the `/setupopenai` command here as you're sharing sensitive information.")
            await ctx.delete()
            return False

    # This slash command allows a user to create their OpenAI API USER file in order to use the /image command.
    @slash_command(name="setupopenai", description="DM Gary your OpenAI API token and Orginization name in order to use `/image`.")
    @commands.check(is_dm)
    async def setup_openai(self, ctx, openai_api_token:  discord.Option(str, required=True), openai_org: discord.Option(str, required=True)):
        result = await create_openai_user(ctx, openai_api_token, openai_org)
        return await ctx.respond(result)

    # This is the slash command to generate images.
    @slash_command(name="image", description="Gary will use AI to generate an image with the given prompt!")
    async def create_image(self, ctx, prompt: Option(str, description="What would you like Gary to create an image of?"), size: discord.Option(str, choices=["Square", "Portrait", "Landscape"], default="Square")):
        openai_user = await get_openai_user(ctx)

        if openai_user == None:
            return await ctx.respond("You need to setup an OpenAI API Token. Then DM Gary and use `/setupopenai <api_token> <org>`\nhttps://openai.com/\n\n**Why does Gary need this?**\n" \
                                     "Gary uses OpenAI's DALLE-3 image generation API to create your amazing ideas. However, this isn't cheap. Providing your own API Token and Orginization " \
                                     "allows users to pay for only the images that they generate. **You can set hard usage limits through OpenAI.**\n\n**How much does this cost?**\nEvery " \
                                     "image generated costs $0.08 or $0.12 per image depending on the size you select during generation.\n\n**How will I be charged?**\nAll payments are made through OpenAI. Gary doesn't read, write, " \
                                     "or store your payment methods 😊", ephemeral=True)
        
        self.users_openai_client = OpenAI(api_key=openai_user["openai_api_token"], organization=openai_user["openai_org"])
        
        # We want to defer this because this operation will almost certainly take longer than 3 seconds.
        await ctx.defer()

        if size == "Square":
            size = "1024x1024"
        elif size == "Portrait":
            size = "1024x1792"
        elif size == "Landscape":
            size = "1792x1024"

        # This is the actual request that generates the image.
        # We need to put it in a try incase the prompt violates ChatGPT's rules. (nudity, gore, etc.)
        try:
            response = self.users_openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="hd",
                n=1
            )
            image_url = response.data[0].url
        except openai.BadRequestError:
            await ctx.author.send("You've reached your billing limit for the month. Adjust your OpenAI accounts billing limit to continue!")
            return await ctx.followup.send("Something went wrong, please check Gary's DM.")
        except openai.AuthenticationError as e:
            await ctx.author.send(f"You're provided API token or Orginization isn't correct. Please double check these and try again.\n```{e}```")
            return await ctx.followup.send("Something went wrong, please check Gary's DM.")
        except Exception as e:
            if "safety system" in str(e):
                await ctx.author.send("Your prompt was rejected by the safety team. Try altering your request!")
                return await ctx.followup.send("Something went wrong, please check Gary's DM.")
            else:
                print(e)
                await ctx.author.send("I couldn't complete the request at this time, please try again!")
                return await ctx.followup.send("Something went wrong, please check Gary's DM.")

        # This converts the image into a useable format we can feed straight into Discord without first saving the file.
        data = await get_image_file(image_url)

        if data == False:
            return await ctx.followup.send("Something went wrong, I couldn't generate the image. Please try again.")
        
        # This gives the image a "file name" incase a user does actually want to download and save the image.
        filename = str(prompt[0:12]).replace(" ", "_")
        file = discord.File(data, f"{filename}.png")
        
        # Here we generate the embed.
        embed = discord.Embed(title=f"{ctx.author.name}'s Image")
        embed.set_image(url=f"attachment://{filename}.png")
        await ctx.followup.send(embed=embed, file=file, view=ImageViewer(bot=self.bot, ctx=ctx, prompt=prompt, size=size))

# Standard bot setup.
def setup(bot):
    bot.add_cog(AskGary(bot))