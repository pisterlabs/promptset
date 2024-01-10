import asyncio
import os
import httpx
import nextcord
from nextcord.ext import commands
import openai
import requests
from gtts import gTTS
from pydub import AudioSegment
import asyncio
import os
## Todo: Add more sub commands (Change whole file)
import speech_recognition as sr
import pyttsx3

from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentType
import os

serper_api_key = os.getenv("SERPAPI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
scenex_api_key = os.getenv("SCENEX_API_KEY")

from langchain import OpenAI, ConversationChain
from langchain.agents import load_tools, initialize_agent

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


# agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True) for voice

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import json
import re
openai_api_key ='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'
serpapi_api_key =os.getenv("SERPAPI_API_KEY")

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
toolkit = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

document_text = None
CHUNK_SIZE = 1024
url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"

headers = {
  "Accept": "audio/mpeg",
  "Content-Type": "application/json",
  "xi-api-key": "2a80772c8a7a5c8dba31d81ea822acd9"
}

data_template = {
  "text": "",
  "voice_settings": {
    "stability": 0,
    "similarity_boost": 0
  }
}


def listen_and_convert_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except Exception as e:
        print("Sorry, could not recognize your voice.")
        return None

def text_to_speech(text):
    # Prepare data for text-to-speech API
    data = data_template.copy()
    data['text'] = text

    # Send request to the text-to-speech API
    response = requests.post(url, json=data, headers=headers)

    # Save the response as an mp3 file
    with open('output.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    # Play the mp3 file using the system's default media player
    import platform
    if platform.system() == "Windows":
        os.startfile("output.mp3")
    elif platform.system() == "Linux":
        os.system(f"xdg-open output.mp3")
    elif platform.system() == "Darwin":
        os.system(f"open output.mp3")

    # Add a delay to ensure the audio finishes playing before deleting the mp3 file
    import time
    time.sleep(len(text) * 0.1)

    # Delete the mp3 file
    os.remove("output.mp3")
    

class WadderChatText(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    @nextcord.slash_command(name="qlangchain")
    async def main(self, interaction: nextcord.Interaction):
        pass
    
    @main.subcommand(name="qsay",description="Get Voice from Text")
    async def wsay(self, interaction: nextcord.Interaction, *, text: str):
        # Generate a response using OpenAI GPT model
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=text,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )

        # Extract the generated text
        generated_text = response.choices[0].text.strip()

        # Prepare data for text-to-speech API
        data = data_template.copy()
        data['text'] = generated_text

        # Send request to the text-to-speech API
        response = requests.post(url, json=data, headers=headers)

        # Save the response as an mp3 file
        with open('output.mp3', 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

        # Connect to the voice channel
        channel = interaction.user.voice.channel
        if not channel:
            await interaction.send("You must be in a voice channel to use this command.")
            return

        vc = await channel.connect()

        # Add a short delay before playing the audio
        await asyncio.sleep(0.5)

        # Play the mp3 file
        vc.play(nextcord.FFmpegPCMAudio(executable="ffmpeg", source="output.mp3"))

        # Wait for the audio to finish playing
        while vc.is_playing():
            await asyncio.sleep(1)

        # Disconnect from the voice channel
        await vc.disconnect()

        # Delete the mp3 file
        os.remove("output.mp3")

    @main.subcommand(name="qspeak",description="Speak with Wadder")
    async def wspeak(self, interaction: nextcord.Interaction):
        
        recognizer = sr.Recognizer()

        # Get the voice channel the user is in
        voice_channel = interaction.user.voice.channel

        if voice_channel is None:
            await interaction.response.send_message("You need to be in a voice channel for me to join.")
            return

        # Join the voice channel
        voice_client = await voice_channel.connect()

        while True:
            # Create a listener for voice commands
            def voice_command_listener(audio_data):
                try:
                    input_text = recognizer.recognize_google(audio_data)
                    print(f"You said: {input_text}")

                    if input_text.strip().lower() == "end":
                        return False, None

                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=input_text,
                        max_tokens=150,
                        n=1,
                        stop=None,
                        temperature=0.5,
                    )

                    generated_text = response.choices[0].text.strip()
                    return True, generated_text

                except sr.UnknownValueError:
                    print("Sorry, I couldn't understand what you said. Please try again.")
                    return True, None

            # Listen for voice commands using the helper function
            with sr.Microphone() as source:
                print("Listening...")
                audio_data = recognizer.listen(source)
                should_continue, generated_text = voice_command_listener(audio_data)

            if not should_continue:
                break

            if generated_text:
                # Prepare data for text-to-speech API
                data = data_template.copy()
                data['text'] = generated_text

                # Send request to the text-to-speech API
                response = requests.post(url, json=data, headers=headers)

                # Save the response as an mp3 file
                with open('output.mp3', 'wb') as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

                # Load the response.mp3 file and play it through the Discord bot
                audio_source = nextcord.FFmpegPCMAudio("output.mp3")
                voice_client.play(audio_source, after=lambda e: print(f"Error: {e}") if e else None)
                await asyncio.sleep(audio_source.duration)

                # Delete the response.mp3 file
                if os.path.exists("output.mp3"):
                    os.remove("output.mp3")

        # Disconnect from the voice channel
        await voice_client.disconnect()

        
    @main.subcommand(name="qsearch", description="Search Google")
    async def search_google(self, interaction: nextcord.Interaction, *, query: str):
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # Initialize the OpenAI language model
        llm = OpenAI(temperature=0)

        # Load tools and initialize the agent
        tools = load_tools(["google-serper"], llm=llm)
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

        # Perform the Google search
        results = agent.run(query)

        # Send the search results in the chat
        response = "Top results:\n"
        for result in results[:3]:
            response += f"{result['title']} - {result['link']}\n"

        await interaction.send(response)    
    

    
    @main.subcommand(name="qchat", description="Chat with a AI Bot")
    async def wchat(self, interaction: nextcord.Interaction, *, message: str):
        response = conversation.predict(input=message)
        await interaction.send(response)
        
    @main.subcommand(name="agents", description="chat with an AI Agent")
    async def agentquery(self, interaction: nextcord.Interaction, *, message: str):
        await interaction.response.defer()

        # Including the provided functionality within agentquery subcommand
        response = conversation.run(message)

        if response:  # Check if the response is not empty
            await interaction.send(response)
        else:
            await interaction.send("An error occurred while processing your request.")

    

def setup(bot):
    bot.add_cog(WadderChatText(bot))
