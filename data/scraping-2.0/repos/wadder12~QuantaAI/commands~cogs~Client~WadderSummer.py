
# ! Successfully summarized the video: https://www.youtube.com/watch?v=5qap5aO4i9A
# Todo: Make it follow the prompt? Could be missing intructions
# ! needs work

import logging
import os
import nextcord # add this
import openai
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nextcord.ext import commands
from pytube import YouTube

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup(bot: commands.Bot):
    bot.add_cog(SummaryCog(bot)) # please put this on bottom lol


def progress_func(chunk=None, file_handle=None, remaining=None):
    """progress call back function for the Summarize function"""
    logger.info("progressing...")


def complete_func(self, path):
    """complete callback function for the Summarize function"""
    logger.info("complete")
    logger.info(self)
    logger.info(path)

async def on_application_command_error(interaction: nextcord.Interaction, error):
    if isinstance(error, commands.CommandInvokeError):
        logger.error(f"In {interaction.command}: {error.original}")
    if not interaction.response.is_done():
        await interaction.send(content="An error occurred while processing the command.")
async def download_yt_file(link):
    yt = YouTube(
        link,
        on_progress_callback=progress_func,
        on_complete_callback=complete_func,
        use_oauth=True,
        allow_oauth_cache=True,
    )
    logger.info("Processing:  " + yt.title)
    stream = yt.streams.filter(only_audio=True).last()
    try:
        ytFile = stream.download(os.getenv("SAVE_PATH"))
        logger.info(f"Processing complete. saving to path {ytFile}")
    except Exception as e:
        ytFile = None
        logger.info(f"Error processing {e}")
    return ytFile


class SummaryCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.is_busy = False 
                            # this is the name     # this is the description
    @nextcord.slash_command(name="summary", description="Summarize a video") # remove commands.commands and add nextcord.slash_command
    async def get_summary(self, interaction: nextcord.Interaction, link): # remove ctx and add interaction: nextcord.Interaction
        await interaction.response.defer()

        ytFile = await download_yt_file(link)

# IN THE WHOLE FILE FIX CTX TO INTERACTION, ANY CTX.AUTHOR TO INTERACTION.USER, AND CTX.SEND TO INTERACTION.REPLY (OR INTERACTION.SEND) DEPENDING ON THE CONTEXT
# DONT USE ALL CAPS, JUST FOR SHOWING YOU WHAT TO CHANGE



        audio_file = open(ytFile, "rb") #
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        transcript = str(transcript)
        logger.info(transcript)
        prompt = f"Please provide the following information based on the transcript:\n" \
         f"1. A title for the transcript in under 15 words.\n" \
         f"2. --Summary--: Write a summary of the provided transcript.\n" \
         f"3. --Additional Info--: Provide a list of main points in the provided transcript.\n" \
         f"4. Provide a list of action items based on the transcript.\n" \
         f"5. Provide a list of follow-up questions related to the transcript.\n" \
         f"6. Provide a list of potential arguments against the information in the transcript.\n" \
         f"For each list, use Heading 2 before writing the list items. Limit each list item to 200 words and return no more than 20 points per list.\n" \
         f"Example output:\n" \
         f"Title: Student Arrested for Series of Attacks\n" \
         f"--Summary--\n" \
         f"Former UC Davis student Carlos Dominguez was arrested for a series of attacks in Davis, California. The attacks began two days after he was forced to leave school for academic reasons.\n" \
         f"--Additional Info--\n" \
         f"1. Carlos Dominguez is 21 years old.\n" \
         f"2. He had no prior arrests.\n" \
         f"3. Students are relieved that the suspect is off the streets.\n" \
         f"Transcript: "

        llm = OpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
        num_tokens = llm.get_num_tokens(transcript)
        await interaction.send(f"Number of Tokens in transcript: {num_tokens}")
        logger.info(f"Number of Tokens in transcript: {num_tokens}")
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
        docs = text_splitter.create_documents([prompt, transcript])
        summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)
        output = summary_chain.run(docs)

        await interaction.send(output)
        return output


def setup(bot: commands.Bot):
    bot.add_cog(SummaryCog(bot))