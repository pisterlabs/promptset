import logging
import discord
from discord.ext import commands
from llama_index import ServiceContext
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings

# Setup logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up the Discord client with updated intents
intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Remove built-in help because we use it already
bot.remove_command("help")

# Set up the LLaMA index service context with a local embedding model
embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# Assuming index is a global variable holding your LLaMA index
# and llm is a global variable holding your local LLM
index = None  # Replace with your LLaMA index instance
llm = None  # Replace with your local LLM instance

async def get_completion(prompt, max_tokens):
    try:
        responses = index.search(prompt, k=5)
        response = llm.generate_response(prompt, responses)
        return response  # Returning the response directly, without truncating
    except Exception as e:
        logging.exception("Exception in get_completion")
        return f"Error: {e}"

async def send_large_message(channel, content, max_length=2000):
    for i in range(0, len(content), max_length):
        await channel.send(content[i:i + max_length])

@bot.event
async def on_ready():
    logging.info(f"{bot.user.name} is now online!")

@bot.command(name="llm")
async def llm(ctx, *, message):
    try:
        response = await get_completion(message, 4096)
        await send_large_message(ctx.channel, f"Response: {response}")
    except Exception as e:
        logging.exception("Exception in !llm command")
        await ctx.send(f"Error: {e}")

if __name__ == "__main__":
    bot.run("")  # Replace with your actual bot token
