
import asyncio
import re

import requests 

#* LangChain Imports
import os
import pprint

import nextcord
#* OpenAI Imports
import openai
import speech_recognition as sr
from langchain import OpenAI
from langchain.agents import AgentType, Tool, initialize_agent, load_tools
from langchain.llms import OpenAI
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import SceneXplainTool
from langchain.utilities import GoogleSerperAPIWrapper
from nextcord.ext import commands

serper_api_key = os.getenv("SERPAPI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
scenex_api_key = os.getenv("SCENEX_API_KEY")
tool = SceneXplainTool()
tools = [tool]


openai.api_base = "https://oai.hconeai.com/v1"
llm = OpenAI(temperature=0, 
             openai_api_key=os.environ['OPENAI_API_KEY'],
             headers={
                    "Helicone-Auth": "Bearer sk-re62dca-taaufcq-r2m44wa-su3niuy"})
toolkit = load_tools(["serpapi"], llm=llm, serpapi_api_key=os.environ["SERPAPI_API_KEY"]) # chaned to SERPAPI_API_KEY from SERPER_API_KEY
agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)

search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]

self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)

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


#call_phrases = {}  # Store call_phrases for each user as {user_id: call_phrase}

class WadderVTV(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        #self.call_phrases = {}  # Store call_phrases for each user as {user_id: call_phrase}
        #self.listen_for_call_phrases.start()  # Start the background task
    
    
    @nextcord.slash_command(name="waddervtv")
    async def main(self, interaction: nextcord.Interaction):
        pass
    
    @main.subcommand(name='voice_search') # Todo: Make it go longer answer questions and use defer to make the command not end or time out
    async def voice_search(self, interaction: nextcord.Interaction):
        await interaction.response.defer()
        
        recognizer = sr.Recognizer()

        # Get the voice channel the user is in
        voice_channel = interaction.user.voice.channel

        if voice_channel is None:
            await interaction.send("You need to be in a voice channel for me to join.")
            return

        # Join the voice channel
        voice_client = await voice_channel.connect()

        while True:
            def voice_command_listener(audio_data):
                try:
                    input_text = recognizer.recognize_google(audio_data)
                    print(f"You said: {input_text}")

                    return input_text.strip().lower()

                except sr.UnknownValueError:
                    print("Sorry, I couldn't understand what you said. Please try again.")
                    return None

            # Listen for voice commands using the helper function
            with sr.Microphone() as source:
                print("Listening...")
                audio_data = recognizer.listen(source)
                input_text = voice_command_listener(audio_data)

            if input_text == "end":
                break

            if input_text:
                # Use the search command with the recognized text as the query
                query = input_text
                response = agent({"input": query})
                response_text = ""
                for step in response["intermediate_steps"]:
                    response_text += f"{step[1]}\n{step[0][2]}\n\n"

                # Use regular expressions to find the final answer
                match = re.search(r'Final Answer: (.+)', response_text)
                if match:
                    answer = match.group(1)
                else:
                    answer = "Sorry, I couldn't find an answer."

                # Prepare data for text-to-speech API
                data = data_template.copy()
                data['text'] = response_text

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



    
    # WolframAlpha API
    #@main.subcommand(name="wolframcalc")
    #async def wolfram_calc(self, interaction: nextcord.Interaction, *, query: str):
        #await interaction.response.defer()
        #try:
            #result = wolfram.run(query)
            #await interaction.send(f"Result for '{query}':\n```\n{result}\n```")
        #except Exception as e:
            #print(f"Error: {e}")
            #await interaction.send(f"Sorry, I couldn't find a result for '{query}'.")
    # 
    # ** Serp Google Search  (Works Very Well, this was a bitch) **
    @main.subcommand(name="serp", description="Perform a Google search")
    async def search_google(self, interaction: nextcord.Interaction, *, query: str):
        # Define the computer animation frames
        animation = [
            "```yaml\n[Performing Google search...     ]```",
            "```yaml\n[Performing Google search...•    ]```",
            "```yaml\n[Performing Google search...••   ]```",
            "```yaml\n[Performing Google search...•••  ]```",
            "```yaml\n[Performing Google search...•••• ]```",
            "```yaml\n[Performing Google search...•••••]```",
            "```yaml\n[Performing Google search... ••••]```",
            "```yaml\n[Performing Google search...  •••]```",
            "```yaml\n[Performing Google search...   ••]```",
            "```yaml\n[Performing Google search...    •]```",
            "```yaml\n[Performing Google search...     ]```",
            "```yaml\n[Performing Google search...    ]```",
            "```yaml\n[Performing Google search...•   ]```",
            "```yaml\n[Performing Google search...••  ]```",
            "```yaml\n[Performing Google search...••• ]```",
            "```yaml\n[Performing Google search...••••]```",
            "```yaml\n[Performing Google search...•••••]```",
            "```yaml\n[Performing Google search...•••• ]```",
            "```yaml\n[Performing Google search...•••  ]```",
            "```yaml\n[Performing Google search...••   ]```",
            "```yaml\n[Performing Google search...•    ]```",
        ]

        # Send the initial loading message
        loading_message = await interaction.response.send_message(animation[0])

        # Animate the loading message
        for frame in animation[1:]:
            await loading_message.edit(content=frame)
            await asyncio.sleep(0.5)

        # Perform the Google search
        llm = OpenAI(temperature=0,openai_api_key=os.environ['OPENAI_API_KEY'])
        tools = load_tools(["google-serper"], llm=llm)
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        result = agent.run(query)
        print("Result:", result)

        # Create an embed to display the search result
        embed = nextcord.Embed(title="Google Search Result", color=nextcord.Color.blue())
        embed.add_field(name="Result", value=f"{result} :mag:", inline=False)  # Add the magnifying glass emoji

        await loading_message.edit(content="Google Search Result", embed=embed)


    
    # ** Give a url image link and have it explain it and what it means 
    # ! Works still
    @main.subcommand(name="scenexplain", description="Explain a scene with a URL image")
    async def scene_explain(self, interaction: nextcord.Interaction, *, image_url: str):
        # Define the computer animation frames
        animation = [
            "```yaml\n[Analyzing the scene...     ]```",
            "```yaml\n[Analyzing the scene...•    ]```",
            "```yaml\n[Analyzing the scene...••   ]```",
            "```yaml\n[Analyzing the scene...•••  ]```",
            "```yaml\n[Analyzing the scene...•••• ]```",
            "```yaml\n[Analyzing the scene...•••••]```",
            "```yaml\n[Analyzing the scene... ••••]```",
            "```yaml\n[Analyzing the scene...  •••]```",
            "```yaml\n[Analyzing the scene...   ••]```",
            "```yaml\n[Analyzing the scene...    •]```",
            "```yaml\n[Analyzing the scene...     ]```",
            "```yaml\n[Analyzing the scene...    ]```",
            "```yaml\n[Analyzing the scene...•   ]```",
            "```yaml\n[Analyzing the scene...••  ]```",
            "```yaml\n[Analyzing the scene...••• ]```",
            "```yaml\n[Analyzing the scene...••••]```",
            "```yaml\n[Analyzing the scene...•••••]```",
            "```yaml\n[Analyzing the scene...•••• ]```",
            "```yaml\n[Analyzing the scene...•••  ]```",
            "```yaml\n[Analyzing the scene...••   ]```",
            "```yaml\n[Analyzing the scene...•    ]```",
        ]

        # Send the initial loading message
        loading_message = await interaction.response.send_message(animation[0])

        # Animate the loading message
        for frame in animation[1:]:
            await loading_message.edit(content=frame)
            await asyncio.sleep(0.5)

        # Initialize the OpenAI language model
        llm = OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
        memory = ConversationBufferMemory(memory_key="chat_history")
        agent = initialize_agent(
            tools, llm, memory=memory, agent="conversational-react-description", verbose=True
        )

        # Use the SceneXplain tool to analyze the image
        output = agent.run(input=f"What is in this image {image_url}? ")

        # Delete the loading message
        await loading_message.delete()

        # Create an embed to display the result
        embed = nextcord.Embed(title="Scene Explanation", color=nextcord.Color.blue())
        embed.add_field(name="Result", value=f"{output} :camera:", inline=False)  # Add the camera emoji

        await interaction.send(embed=embed)


        
        
    # ** All Search Types work, but the web search does not. Make all embeds
    @main.subcommand(name="searchall", description="Perform different types of searches")
    async def search(self, interaction: nextcord.Interaction, search_type: str, *, query: str):
        # Define the computer animation frames
        animation = [
            "```yaml\n[Performing search...     ]```",
            "```yaml\n[Performing search...•    ]```",
            "```yaml\n[Performing search...••   ]```",
            "```yaml\n[Performing search...•••  ]```",
            "```yaml\n[Performing search...•••• ]```",
            "```yaml\n[Performing search...•••••]```",
            "```yaml\n[Performing search... ••••]```",
            "```yaml\n[Performing search...  •••]```",
            "```yaml\n[Performing search...   ••]```",
            "```yaml\n[Performing search...    •]```",
            "```yaml\n[Performing search...     ]```",
            "```yaml\n[Performing search...    ]```",
            "```yaml\n[Performing search...•   ]```",
            "```yaml\n[Performing search...••  ]```",
            "```yaml\n[Performing search...••• ]```",
            "```yaml\n[Performing search...••••]```",
            "```yaml\n[Performing search...•••••]```",
            "```yaml\n[Performing search...•••• ]```",
            "```yaml\n[Performing search...•••  ]```",
            "```yaml\n[Performing search...••   ]```",
            "```yaml\n[Performing search...•    ]```",
        ]

        # Send the initial loading message
        loading_message = await interaction.response.send_message(animation[0])

        # Animate the loading message
        for frame in animation[1:]:
            await loading_message.edit(content=frame)
            await asyncio.sleep(0.5)

        # Perform the search based on the search type
        if search_type.lower() == 'web':
            search = GoogleSerperAPIWrapper(type=search_type.lower())
            result = search.run(query)
            await loading_message.edit(content=result)
        elif search_type.lower() == 'self_ask_with_search':
            result = self_ask_with_search.run(query)
            await loading_message.edit(content=result)
        elif search_type.lower() in ['images', 'news', 'places']:
            search = GoogleSerperAPIWrapper(type=search_type.lower())
            results = search.results(query)
            pprint.pp(results)
            if search_type.lower() == 'images':
                await loading_message.edit(content='\n'.join([result['link'] for result in results['images'][:5]]))
            elif search_type.lower() == 'news':
                await loading_message.edit(content='\n'.join([result['link'] for result in results['news'][:5]]))
            elif search_type.lower() == 'places':
                places_info = [f"{place['title']} - {place['address']}\nRating: {place['rating']}, Category: {place['category']}" for place in results['places'][:5]]
                await loading_message.edit(content='\n\n'.join(places_info))
        else:
            await loading_message.edit(content='Invalid search type. Please choose from "web", "self_ask_with_search", "images", "news", or "places".')

            
def setup(bot):
    bot.add_cog(WadderVTV(bot))