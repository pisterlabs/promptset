import json
import openai

from discord import Intents, File
from discord.ext import commands

from input_handler import InputHandler
from hallucination import Hallucination
from image_server_client import ImageServerClient


class DiscordBot:
    def __init__(self, discord_auth_token, openai_api_token, image_server_host, image_server_port):
        intents = Intents.default()
        intents.message_content = True
        self.token = discord_auth_token
        openai.api_key = openai_api_token
        self.image_server = ImageServerClient(image_server_host, image_server_port)
        self.bot = commands.Bot(command_prefix='/',
                                intents=intents,
                                description='A bot that generates images from text prompts using the Stable Diffusion '
                                            'model.',
                                help_command=None)

        @self.bot.command(name='dream', help='Generates an image given a prompt')
        async def generate(ctx, *, message):
            await self.mark_message_processing(ctx)

            input_handler = InputHandler()
            prompt, options = input_handler.sanitize_input(message)
            await self.generate_image(ctx, prompt, ctx.message.author.id, options)

        @self.bot.command(name='hallucinate', help='Generates a hallucinated image given a prompt')
        async def generate(ctx, *, message):
            await self.mark_message_processing(ctx)

            input_handler = InputHandler()
            prompt, options = input_handler.sanitize_input(message)

            temperature = 0.9
            if 'lucid' in options:
                lucid = max(0.0, min(float(options['lucid']), 1.0))
                temperature = 1.0 - lucid

            negative_prompt = ""
            if 'negative_prompt' in options:
                negative_prompt = options['negative_prompt']

            hallucination = Hallucination(prompt, negative_prompt, temperature)
            transformed_prompt, transformed_negative_prompt, explanation = hallucination.run()
            await ctx.message.reply(f'We are coming out of the deep dream.\n'
                                    f'**Prompt:** {transformed_prompt}\n'
                                    f'**Negative:** {transformed_negative_prompt}\n'
                                    f'**Explanation:** {explanation}')

            options['negative_prompt'] = transformed_negative_prompt

            await self.generate_image(ctx, transformed_prompt, ctx.message.author.id, options)

    def run(self):
        self.bot.run(self.token)

    async def generate_image(self, ctx, prompt, actor_id, options):
        websocket = await self.image_server.send_request(prompt, actor_id, options)

        status_message = await ctx.send('Image generation started...')

        async for message in websocket:
            if message.startswith("Progress:"):
                progress = float(message.split(":")[1].strip().strip("%"))
                await status_message.edit(content=f'Image generation progress: {progress:.2f}%')
            else:
                # Here I assume that the received message is a serialized list (JSON)
                response = json.loads(message)
                image_paths = response['image_paths']
                options = response['options']

                await status_message.edit(content='Image generation completed!')
                for image_path in image_paths:
                    options_string = " ".join(f'--{k} {v}' for k, v in options.items() if k != 'lucid')
                    repro_str = f'{prompt} {options_string}'
                    await ctx.message.reply(
                        file=File(image_path),
                        mention_author=True,
                        content=f'**Serving up a hot new image fresh out of the oven!** \n'
                                f'`id: {image_path}`\n'
                                f'`seed: {options["seed"]}`\n'
                                f'`repro: {repro_str}`\n'
                    )

        await ctx.message.remove_reaction('⏳', self.bot.user)
        await ctx.message.add_reaction('✅')
        await status_message.delete()

        await websocket.close()

    async def mark_message_processing(self, ctx):
        await ctx.message.add_reaction('🦑')
        await ctx.message.add_reaction('⏳')
