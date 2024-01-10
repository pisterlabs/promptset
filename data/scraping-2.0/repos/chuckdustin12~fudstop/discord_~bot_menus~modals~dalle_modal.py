# At the top of the file.
import disnake
from disnake.ext import commands
from disnake import TextInputStyle
import aiohttp
import os
import asyncio
import openai
# Subclassing the modal.
class DalleModal(disnake.ui.Modal):
    def __init__(self,bot):
        self.bot=bot
        # The details of the modal, and its components
        components = [
            disnake.ui.TextInput(
                label="Image Prompt",
                placeholder="A picture of a fork spooning a knife with butter dripping all over it..",
                custom_id="name",
                style=TextInputStyle.paragraph,
                max_length=250,
            ),

        ]
        super().__init__(title="Image Prompt:", components=components)

    # The callback received when the user input is completed.
    async def callback(self, inter: disnake.ModalInteraction):
        await inter.response.defer(with_message=True, ephemeral=True)
        embed = disnake.Embed(title="Dalle3 Image Result")
        user_input = inter.text_values
        # Wait for a message from the same user

        
        # Extract the actual prompt for the image
        image_prompt = user_input['name']

        # Call the OpenAI API to generate the image
        response = openai.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            n=1,
            size="1024x1024"
        )

        # Get the URL of the generated image
        image_url = response.data[0].url

        # Download the image using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    await inter.send("Failed to download the image.")
                    return
                data = await resp.read()

        # Save the image to a file
        file_name = "generated_image.png"
        with open(file_name, "wb") as file:
            file.write(data)

        # Send the image file to the chat
        with open(file_name, "rb") as file:
            await inter.send(file=disnake.File(file, file_name))

        embed.set_image(f"attachment://{file_name}")

        await inter.edit_original_message(embed=embed)