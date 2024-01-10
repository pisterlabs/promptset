from discord import Intents
from discord.ext import commands
from langchain.chat_models import ChatOpenAI

def get_discord_client(intents:Intents, openai_model:str, model_temp:float) -> commands.Bot:
    client = commands.Bot(command_prefix = '!', intents=intents)

    client.video_db = None
    client.openaiChat = ChatOpenAI(model_name=openai_model, temperature=model_temp)
    client.template = """
            You are a helpful assistant that that can answer questions about youtube videos 
            based on the video's transcript: {docs}
            and video meta informations: {meta_info}
            
            Only use the factual information from the transcript or meta informations to answer the question.
            
            If you feel like you don't have enough information to answer the question, say "I don't know".
            
            Your answers should be verbose and detailed.
            You can use bullet points to format the answer
            """
    return client