import json
import re
import openai
import requests
from discord.ext import commands
from prompt import prompt
import concurrent.futures

class AsmCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command(name="asm", description="ASM Bot répond à ta question sur l'ASM ou la mémoire")
    async def asm(self, ctx, *, user_query):
        print(":", user_query)
        
        # Send a message in the text channel where the command was invoked
        await ctx.respond(f"{ctx.author.mention} a demandé: {user_query}")

        system_prompt = prompt
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
        }
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            "max_tokens": 6000,
        }

        async with ctx.typing():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                response = await self.bot.loop.run_in_executor(
                    executor, self.get_response, headers, data
                )

            if response.status_code == 200:
                response_json = response.json()
                try:
                    response_message = response_json["choices"][0]["message"]["content"]
                    await send_large_message(ctx, response_message)
                    print("Completion:---------------------------------------------------------------------\n", response_message)
                    print("Full API Response:--------------------------------------------------------------\n", json.dumps(response_json, indent=2))
                except Exception as e:
                    print(f"Error sending message: {e}")
            elif response.status_code == 429:
                await ctx.send("Aïe, je reçois trop de requêtes, réessaye 🤕")
            else:
                print(f"Error: API returned status code {response.status_code}")
                await ctx.send(f"Erreur code {response.status_code}, veuillez contacter @kitsuiwebster sur Discord.")

    def get_response(self, headers, data):
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
        )
        return response

async def send_large_message(ctx, message):
    message_parts = []
    current_part = ""
    sentences = re.split(r'(?<=[.!?])\s+', message)
    
    for sentence in sentences:
        if len(current_part) + len(sentence) <= 2000:
            current_part += sentence
        else:
            message_parts.append(current_part)
            current_part = sentence
            
    if current_part:
        message_parts.append(current_part)
        
    for part in message_parts:
        await ctx.send(part)

def setup(bot):
    bot.add_cog(AsmCog(bot))
