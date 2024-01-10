import nextcord
from nextcord.ext import commands
import os
import random
import re
from collections import defaultdict, Counter
import json
import requests
from bs4 import BeautifulSoup



import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
class GPT3Model:
    def __init__(self):
        pass

    def generate_response_gpt3(self, prompt, conversation_history):
        formatted_history = ""
        for i, (msg_type, msg_content) in enumerate(conversation_history):
            sender = "User" if msg_type == "user" else "Waddermind"
            formatted_history += f"{sender}: {msg_content}\n"

        formatted_prompt = f"{formatted_history}User: {prompt}\nWaddermind:"

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=formatted_prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

class WadderMindML:
    def __init__(self):
        self.words = defaultdict(Counter)
        self.previous_word = defaultdict(str)
        self.total_messages = 0
        self.min_messages = 100  # Set this to the minimum number of messages you want the chatbot to learn from

    def learn(self, message):
        words = re.findall(r'\b\w+\b', message.lower())
        for i, word in enumerate(words):
            if i > 0:
                self.words[self.previous_word[word]][word] += 1
            self.previous_word[word] = word
        self.total_messages += 1

    def generate_response(self, prompt):
        response = []
        current_word = random.choice(prompt.split())

        for _ in range(10):
            if not self.words[current_word]:
                break
            next_word = random.choices(list(self.words[current_word].keys()), weights=self.words[current_word].values())[0]
            response.append(next_word)
            current_word = next_word

        return ' '.join(response)
    
    def has_learned_enough(self):
        return self.total_messages >= self.min_messages
    
    
    def load_data(self, server_id):
        file_name = f'data/data_{server_id}.json'

        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                data = json.load(f)
                self.words = defaultdict(Counter, {k: Counter(v) for k, v in data['words'].items()})
                self.previous_word = defaultdict(str, data['previous_word'])
                self.total_messages = data['total_messages']

    def save_data(self, server_id):
        file_name = f'data/data_{server_id}.json'

        data = {
            'words': {k: dict(v) for k, v in self.words.items()},
            'previous_word': dict(self.previous_word),
            'total_messages': self.total_messages
        }

        with open(file_name, 'w') as f:
            json.dump(data, f)
            
            
            
class HybridModel(WadderMindML, GPT3Model):
    def __init__(self):
        super().__init__()
        self.conversation_history = []

    def add_message_to_history(self, message_type, message_content):
        self.conversation_history.append((message_type, message_content))

    def generate_response(self, prompt):
        if self.has_learned_enough():
            return super().generate_response(prompt)
        else:
            return self.generate_response_gpt3(prompt, self.conversation_history)

class ChatGPTServerCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.chatbots = {}
        

    def learn(self, server_id, message):
        if server_id not in self.chatbots:
            self.chatbots[server_id] = WadderMindML()
            self.chatbots[server_id].load_data(server_id)

        self.chatbots[server_id].learn(message.content)
        self.chatbots[server_id].save_data(server_id)

    def generate_response(self, server_id, prompt):
        return self.chatbots[server_id].generate_response(prompt)

    @commands.Cog.listener()
    async def on_ready(self):
        print("ChatGPTServerCog is ready")
        for guild in self.bot.guilds:
            server_id = str(guild.id)
            if server_id not in self.chatbots:
                self.chatbots[server_id] = HybridModel()
                self.chatbots[server_id].load_data(server_id)

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot or message.content.startswith('!'):
            return

        server_id = str(message.guild.id)
        if server_id not in self.chatbots:
            self.chatbots[server_id] = HybridModel()

        # Check if the message has an attachment
        if message.attachments:
            # Process the attachment if it's a text file
            for attachment in message.attachments:
                if attachment.filename.endswith('.txt'):
                    content = await attachment.read()
                    text_data = content.decode('utf-8')
                    self.chatbots[server_id].learn(text_data)
                    self.chatbots[server_id].save_data(server_id)
        else:
            # Check if the message has a URL link
            if message.content.startswith(('http://', 'https://')):
                # Get the HTML content of the URL
                response = requests.get(message.content)
                content = response.content
            else:
                content = message.content
            # Parse the message content
            soup = BeautifulSoup(content, 'html.parser')
            links = soup.find_all('a')
            for link in links:
                # Parse the link URL
                url = link.get('href')
                # Check if the link is a text file
                if url.endswith('.txt'):
                    # Download the file
                    response = requests.get(url)
                    text_data = response.content.decode('utf-8')
                    self.chatbots[server_id].learn(text_data)
                    self.chatbots[server_id].save_data(server_id)  # Save the data to the JSON file
            # Add the message to the chat history
            self.chatbots[server_id].add_message_to_history("user", message.content)
            self.learn(server_id, message)


    @nextcord.slash_command(name="waddermind")
    async def main(self, interaction: nextcord.Interaction):
        pass
    
    
    @main.subcommand(name="ask", description="Ask Waddermind a question")
    async def ask_question(self, interaction: nextcord.Interaction, *, question):
        server_id = str(interaction.guild.id)
        if server_id not in self.chatbots or not self.chatbots[server_id].has_learned_enough():
            await interaction.send("I haven't learned enough yet. Please give me more time.")
            return

        response = self.generate_response(server_id, question)
        self.chatbots[server_id].add_message_to_history("bot", response)
        await interaction.send(response)
        
        
    @main.subcommand(name="find_channel", description="Find a channel by name")
    async def find_channel(self, interaction: nextcord.Interaction, *, channel_name: str):
        channel = nextcord.utils.get(interaction.guild.channels, name=channel_name.lower())
        if channel:
            await interaction.send(f'The channel "{channel_name}" is {channel.mention}.')
        else:
            await interaction.send(f'No channel found with the name "{channel_name}".')

def setup(bot):
    bot.add_cog(ChatGPTServerCog(bot))

