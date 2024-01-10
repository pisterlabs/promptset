import io
import nextcord
from nextcord import Interaction, Embed, PermissionOverwrite
import openai
import os

import requests

def setup(bot):

    @bot.slash_command()
    async def main(interaction: Interaction):
        pass

    @main.subcommand(description="Chat with an AI bot")
    async def chatbot(interaction: Interaction):
        await interaction.response.defer()

        # Set up the OpenAI API
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Create a private channel for the user
        overwrites = {
            interaction.guild.default_role: PermissionOverwrite(read_messages=False),
            interaction.user: PermissionOverwrite(read_messages=True)
        }
        private_channel = await interaction.guild.create_text_channel(f"chatbot-{interaction.user}", overwrites=overwrites)

        # Begin the chat loop
        while True:
            # Prompt the user for a question or message for the chatbot
            embed = Embed(title="Chatbot", description="What would you like to ask the chatbot? (Enter 'quit' to exit)", color=0x00ff00)
            await private_channel.send(embed=embed)

            # Wait for the user to enter their question or message
            def check(m):
                return m.author == interaction.user and m.channel == private_channel and m.content != ""

            user_input = await bot.wait_for('message', check=check)

            # Check if the user wants to quit
            if user_input.content.lower() == "quit":
                break

            # Generate a response using OpenAI
            response = openai.Completion.create(
                engine="text-davinci-003",  # The model to use
                prompt=user_input.content,  # The user's question or message
                temperature=0.5,  # How "creative" the response should be
                max_tokens=512,  # The maximum number of tokens (words) in the response
                n=1,  # The number of responses to generate
                stop=None,  # Stop generating responses when one of these strings is encountered
            )

            # Send the response to the user in the private channel
            response_text = response.choices[0].text
            print(f"Sending response to channel {private_channel}: {response_text}")
            embed = Embed(title="Chatbot", description=response_text, color=0x00ff00)
            await private_channel.send(embed=embed)

        # Send a goodbye message when the user is done
        embed = Embed(title="Chatbot", description="Goodbye!", color=0x00ff00)
        await private_channel.send(embed=embed)

        # Delete the private channel
        await private_channel.delete()

    @main.subcommand(description="Generate a product name")
    async def generate_product_name1(interaction: Interaction):
        await interaction.response.defer()
        await interaction.followup.send("What's the product prompt?", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        prompt = response.content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Generate a unique product name based on the following prompt:\n{prompt}\nProduct name:",
            max_tokens=1024,
            temperature=0.7,
        )
        product_name = response.choices[0].text.strip()

        # Create an embed object and set its properties
        embed = nextcord.Embed(title="Product Name Generated", description=f"Here's your product name:", color=0x00ff00)
        embed.add_field(name="Prompt", value=prompt, inline=False)
        embed.add_field(name="Product Name", value=product_name, inline=False)

        # Send the embed as a follow-up message
        await interaction.followup.send(embed=embed)

    @main.subcommand(description="code generator")
    async def generate_code1(interaction: nextcord.Interaction):
        """
        Generate code snippets using the Davinci 003 engine.
        """
        await interaction.response.defer()
        await interaction.followup.send("What's the problem description?", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        code_description = response.content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"{code_description}\nCode:",
            max_tokens=1024,
            temperature=0.7,
        )
        code_snippet = response.choices[0].text.strip()
        await interaction.followup.send(f"Here's some code that solves the problem:\n```{code_snippet}```")

    @main.subcommand(description="edit text")
    async def edit(interaction: nextcord.Interaction, text: str):
        # Use the OpenAI API to edit the text
        response = openai.Edit.create(model="text-davinci-edit-001", input=text, instruction="Fix the spelling mistakes")
        edited_text = response["choices"][0]["text"]
        
        # Create an embed with the edited text
        embed = nextcord.Embed(title="Edited Text", description=edited_text, color=0x00ff00)
        
        # Send the embed as a reply to the user's interaction
        await interaction.response.send_message(embed=embed, ephemeral=True)

    
    @main.subcommand(description="generate lyrics")
    async def generate_lyrics(interaction: nextcord.Interaction):
        """
        Generate song lyrics using the Davinci 003 engine.
        """
        await interaction.response.defer()
        await interaction.followup.send("What's the song prompt?", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        prompt = response.content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Write lyrics for a song based on the following prompt:\n{prompt}\nLyrics:",
            max_tokens=1024,
            temperature=0.7,
        )
        lyrics = response.choices[0].text.strip()
        embed = nextcord.Embed(title="Generated Lyrics", description=lyrics, color=0x00ff00)
        await interaction.followup.send(embed=embed)


    @main.subcommand(description="generate a poem")
    async def generate_poem(interaction: nextcord.Interaction):
        """
        Generate a poem using the Davinci 003 engine.
        """
        await interaction.response.defer()
        await interaction.followup.send("What's the poem prompt?", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        prompt = response.content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Write a poem based on the following prompt:\n{prompt}\nPoem:",
            max_tokens=1024,
            temperature=0.7,
        )
        poem = response.choices[0].text.strip()
        await interaction.followup.send(f"Here's your poem:\n```{poem}```")
        
        
        
    @main.subcommand(description="generate a technical documentation")
    async def generate_technical_documentation(interaction: nextcord.Interaction):
        """
        Generate technical documentation using the Davinci 003 engine.
        """
        await interaction.response.defer()
        await interaction.followup.send("Please provide the name of the software program or system:", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        software_name = response.content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Generate technical documentation for the {software_name} software program or system:",
            max_tokens=2048,
            temperature=0.7,
        )
        document = response.choices[0].text.strip()
        embed = nextcord.Embed(title="Generated Technical Documentation", description=document, color=0x00ff00)
        await interaction.followup.send(embed=embed)
        
        
    @main.subcommand(description="summarize text")
    async def summarize(interaction: nextcord.Interaction, text: str):
        """
        Summarize text using the Davinci 003 engine.
        """
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Please summarize the following text:\n{text}\nSummary:",
            max_tokens=60,
            temperature=0.7,
        )
        summary = response.choices[0].text.strip()
        embed = nextcord.Embed(title="Summary", description=summary, color=0x00ff00)
        await interaction.response.send_message(embed=embed)
        
        
    @main.subcommand(description="generate an inspiring quote")
    async def generate_inspiring_quotes(interaction: nextcord.Interaction, topic: str):
        """
        Generate an inspiring quote on a specified topic using the Davinci 003 engine.
        """
        await interaction.response.defer()
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Create an inspiring quote about {topic}:",
            max_tokens=100,
            temperature=0.7,
            n=1,
            stop=None,
        )
        quote = response.choices[0].text.strip()
        embed = nextcord.Embed(title=f"Inspiring Quote on {topic.capitalize()}", description=quote, color=0x00ff00)
        await interaction.followup.send(embed=embed) 
        
        
        
    @main.subcommand(description="generate an image")
    async def generate_images(interaction: nextcord.Interaction):
        """
        Generate images using the Dall-E 2 engine.
        """
        await interaction.response.defer()
        await interaction.followup.send("What do you want to generate an image of?", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        prompt = response.content
        await interaction.followup.send("How many images do you want to generate?", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        n = int(response.content)
        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size="1024x1024"
        )
        images = response['data']
        for image in images:
            with io.BytesIO() as image_binary:
                # Download the image data and write it to a BytesIO buffer
                response = requests.get(image['url'])
                image_binary.write(response.content)
                # Reset the buffer position to the beginning
                image_binary.seek(0)
                # Send the image file to the Discord channel
                await interaction.followup.send(file=nextcord.File(image_binary, filename='generated_image.png'))
                
                
                
        @main.subcommand(description="generate an image")
        async def generate_financial_advice(interaction: nextcord.Interaction):
            """
            Generate personalized financial advice using the Davinci 003 engine. This is for fun not to be used for real life!
            """
            await interaction.response.defer()
            await interaction.followup.send("Please provide your financial data:", ephemeral=True)
            response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
            financial_data = response.content
            
            # Generate financial advice using OpenAI's API
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Given the following financial data:\n{financial_data}\nProvide personalized financial advice:",
                max_tokens=1024,
                temperature=0.7,
            )
            advice = response.choices[0].text.strip()
            
            # Send the financial advice in a nice embed
            embed = nextcord.Embed(
                title="Personalized Financial Advice",
                description=advice,
                color=nextcord.Color.blue()
            )
            embed.set_footer(text="This advice is for fun and should not be taken as professional financial advice.")
            await interaction.followup.send(embed=embed) 
            
            
            
        