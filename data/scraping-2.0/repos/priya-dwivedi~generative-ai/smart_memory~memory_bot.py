import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import asyncio
import json
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from rich.console import Console
from rich.prompt import Prompt

# Create a console object
console = Console()

class Chatbot:
    def __init__(self):
        ## reading config file
        self.client = OpenAI()
        self.messages = []
        self.messages.append({"role": "system", "content":"You are a friendly chatbot who likes to chat with users and extract relevant information. You respond back in JSON format."})
        self.memory = {}

    def set_user_prompt(self):
        user_prompt = f'''
            Chat with the user. If they share personal information like their or family members birthday, or hobbies, likes or dislikes, then extract those too. Store these with entity name and the information in third person style.
            If no personal information is shared, then return None for relevant information.
            I would like output in JSON format. See example below:
            \n Example:
            Query: " I am turning 40 on Dec 19th. I am not sure what to buy. I hope you can make some suggestions"
            Answer:
            {{"response" : "Glad to hear that! I am happy to help. What kind of activities do you enjoy?"}},
            "relevant_info" : [{{'entity': 'user', "information": 'Turning 40 on Dec 19"}}]

            \n Example:
            Query: " My sister loves making cakes! Maybe she can make a chocolate lava cake for me. I would like that"
            Answer:
            {{"response" : "Ohh nice!"}},
            "relevant_info" : [{{'entity': 'user', "information": 'Loves chocolate lava cake"}}, {{'entity': 'sister', "information": 'Likes baking cakes"}}]

            Now respond to user's chat below:
            User: {self.chat}
            Answer: {{'response': "", 'relevant_info':""}}

            '''
        return user_prompt
    
    def collect_memory(self, record):
        for each in record:
            entity = each['entity']
            info = each['information']
            if entity in self.memory.keys():
                self.memory[entity] += ". "+ info
            else:
                self.memory[entity] = ''
                self.memory[entity] += ". "+ info

    async def return_memory(self):
        yield self.memory

    async def call_open_ai(self, chat):
        self.chat = chat
        user_prompt = self.set_user_prompt()
        self.messages.append({"role": "user", "content": user_prompt})
        completion = self.client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=self.messages,
        response_format={"type": "json_object"},
        temperature=0.4,
        )
        result = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": result})
        result = json.loads(result)
        if result['relevant_info'] != "None":
            ## Save this
            self.collect_memory(result['relevant_info'])

        yield {
                "response":result["response"],
                "memory":result["relevant_info"]
            }


if __name__=="__main__":
    
    bot = Chatbot()

    async def CLI():
        mssg_cnt = 0
        while True:
            user_input = Prompt.ask("[bold green] Query [/bold green] ")
            async for output in bot.call_open_ai(user_input):
                if output['response'] !='None':
                    console.print(f"[bold green] 🤖 AI: {output['response']} [/bold green]")
                if output['memory'] != 'None':
                    console.print(f"[bold yellow] 🤖 Add to memory: {output['memory']} [/bold yellow]")
                mssg_cnt +=1

            if mssg_cnt %5 ==0:
                async for output in bot.return_memory():
                    console.print(f"[bold red] 🤖 My memory so far: {output} [/bold red]")

    
    asyncio.run(CLI())

