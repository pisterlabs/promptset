import discord
import os
import openai
import openai.error
from dotenv import load_dotenv

# https://stackoverflow.com/questions/55462226/how-can-i-keep-a-python-script-on-a-remote-server-running-after-closing-out-of-s

load_dotenv('.env_story_teller')


OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
BOT_TOKEN = os.getenv('BOT_TOKEN')

openai.api_key = OPEN_AI_KEY


system = {"role": "system", "content": "You are an expert illustrator for storytelling. You know precisely when something important has happened and are able to create an image prompt from the conversation to illustrate what was said. Something important in the conversation should be something that brings a lot of excitement to the conversation or is something unexpected. As a storyteller, you should know when critical points of the conversation have happened and write the prompts for those. If nothing interesting happened, then return \"STOP\". Remember you must ONLY respond with \"STOP\" or an image prompt description of the conversation, nothing else!\n\nWhen responding with the image prompt, you will become a DALLE-2 prompt generation tool that will generate a suitable prompt for me to generate a picture story based on the story that you have been given, generate a prompt that gives the DALLE-2 AI text to generate a picture model, please narrate in English. To create an effective prompt for DALL·E 2 to generate a good image, follow these guidelines: \n1. Be specific: Clearly describe the elements you want in the image. Include essential details such as colors, shapes, sizes, and positions. \n2. Use descriptive adjectives to provide more context, emotion, or atmosphere in your prompt. This can aid DALL·E 2 in understanding your desired visual outcome. \n3. Combine ideas: You can combine multiple concepts or themes for a more creative and unique image. However, ensure that they are coherent and connected. 4. Avoid ambiguity: Ensure your description is unambiguous, so DALL·E 2 doesn't get confused about what you want in the image. \n5. Keep it concise while describing your requirements, but don't compromise on clarity. Too much information may confuse DALL·E 2, while too little may result in a generic image. For example, if you want an image of a serene, moonlit beach, you could prompt DALL·E 2 with: \"A tranquil beach at night with a full moon reflecting on the water, surrounded by palm trees gently swaying in the breeze.\" Feel free to experiment and refine your prompt as needed.\n6. Do not add any text other than the prompt, such as \"Image Prompt:\" at the beginning of the prompt.\n\nRemember you must ONLY respond with \"STOP\" or a DALL·E 2 image prompt description of the conversation, nothing else! Do NOT add \"Image Prompt:\" to the beginning of the sentence.\nRemember to be listening to the exciting points in the conversation. These moments should be special. Be sure to take your time when thinking about it."}

MAX_MESSAGE_HISTORY = 20

GENERATION_FREQUENCY = 10

class MyClient(discord.Client):

    async def on_ready(self):
        print(f'Logged on as {self.user}!')
        print(system['content'])

        self.channel_counter = dict()

    def generate_chat_gpt_response(self, message_history):
        try:
            messages = []

            messages.append(system)
            messages.extend(message_history[-MAX_MESSAGE_HISTORY:])

            print("Generating ChatGPT response from", messages)
            completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=
                                                    messages,
                                                    temperature=1,
                                                    max_tokens=100,
                                                    top_p=1,
                                                    frequency_penalty=0,
                                                    presence_penalty=0,
                                                    stop=["STOP"]
                                                    )
                                                    
            
            chat_response = completion.choices[0].message.content
            print(chat_response)

            if chat_response is None or chat_response == '' or chat_response == "STOP":
                print("Do not respond with anything")
                chat_response = None
            else:
                if chat_response.startswith("Image Prompt:"):
                    chat_response = chat_response[len('Image Prompt:'):]

        except openai.error.OpenAIError as e:
            print("Rate Limit Reached")
            print(e.http_status)
            print(e.error)
            chat_response = None

        return chat_response
    
    def generate_dalle_output(self, image_prompt):
        try:
            print("Generating DallE image from prompt", image_prompt)
            response = openai.Image.create(
                                            prompt=image_prompt,
                                            n=1,
                                            size="1024x1024"
                                            )
            image_url = response['data'][0]['url']
            print(image_url)
        except openai.error.OpenAIError as e:
            print("Rate Limit Reached")
            print(e.http_status)
            print(e.error)
            image_url = None

        return image_url
    async def on_message(self, message):
        if message.author.bot:
            return

        channel = message.channel

        if not channel.id in self.channel_counter:
            self.channel_counter[channel.id] = 0
        
        self.channel_counter[channel.id] += 1

        print("Number of channel messages", self.channel_counter[channel.id])

        if not self.channel_counter[channel.id] % GENERATION_FREQUENCY == 0:
            return
        
        print("Looking to generate")

        channel_messages = []

        async for history_message in channel.history(limit=MAX_MESSAGE_HISTORY):
            if not history_message.author.bot:
                channel_messages.insert(0, {'role':'user', "content": history_message.content})

        image_prompt = self.generate_chat_gpt_response(channel_messages)

        if image_prompt is not None:
            async with message.channel.typing(): 
                image_url = self.generate_dalle_output(image_prompt)

                if image_prompt is not None:
                    await message.channel.send(image_prompt)
                    await message.reply(image_url)
                    # await message.channel.send(image_url)


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True

    client = MyClient(intents=intents)
    client.run(BOT_TOKEN)
