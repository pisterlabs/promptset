import asyncio
import glob
import os
import traceback

import aiogram.types






from config import bot, dp


#from read_all_files import read_file



#import pickle

from tgbot import get_chat_data

redis=None
async def mem_init():
    if True:
        return #deprecated function
    global redis
    import aioredis
    redis = await aioredis.from_url("redis://localhost")
    if not os.path.exists('data/'):
        os.mkdir('data')





def get_index(chat_id,files=None):
    from llama_index import SimpleDirectoryReader, ServiceContext, SummaryIndex
    documents = SimpleDirectoryReader(input_files=files).load_data()
    index = SummaryIndex.from_documents(documents)
    return index
def smart_youtube_reader(video_url,query_text,model='gpt-3.5-turbo'):
    from llama_index.readers import YoutubeTranscriptReader
    from llama_index.llms import OpenAI
    from llama_index import SimpleDirectoryReader, ServiceContext, SummaryIndex
    reader=YoutubeTranscriptReader()
    documents=reader.load_data([video_url])
    #vector_index = VectorStoreIndex.from_documents(documents)
    llm = OpenAI(temperature=0, model=model)
    service_context = ServiceContext.from_defaults(llm=llm)

    # build summary index
    summary_index = SummaryIndex.from_documents(
        documents, service_context=service_context
    )
    # define query engines
    #vector_query_engine = vector_index.as_query_engine()
    list_query_engine = summary_index.as_query_engine()

    if not query_text:
        query_text='Summarise the main points in a list format on russian language'
    results=list_query_engine.query(query_text)
    print('youtube sum%: ',results.response)
    return results


def get_youtube_transcript(video_url):
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api import NoTranscriptFound

    # using the srt variable with the list of dictionaries
    # obtained by the .get_transcript() function
    id=video_url.split('/')[-1]
    if '?' in id:
        id=id.split('v=')[-1]
    srt=None
    try:
        srt = YouTubeTranscriptApi.get_transcript(id,languages=['ru'])
    except NoTranscriptFound:
        pass
    if not srt:
        try:
            srt = YouTubeTranscriptApi.get_transcript(id)
        except:
            traceback.print_exc()
            srt=[{'text':traceback.format_exc()}]

    content=[s['text'] for s in srt]
    return content

async def query_index(chat_id, query_text,files,model='gpt-3.5-turbo'):
    from llama_index.llms import OpenAI
    def non_async():

        index=get_index(chat_id,files)
        llm = OpenAI(temperature=0, model=model)
        service_context = ServiceContext.from_defaults(llm=llm)
        query_engine = index.as_query_engine(service_context=service_context)
        results = query_engine.query(query_text)
        return results
    results=await asyncio.get_running_loop().run_in_executor(None,non_async)

    return f'{results}'
from aiogram import types
@dp.message_handler(lambda m:m.caption or m.text,content_types=aiogram.types.ContentTypes.DOCUMENT)
async def handle_doc_query(message: types.Message):
    try:
        chat_id = message.chat.id
        text = message.caption or message.text

        # Если прикреплен файл, сделаем из него индекс (пример)

        file_path = await bot.download_file_by_id(message.document.file_id, destination_dir=f"data/{chat_id}")
        # Здесь загрузите файл в индекс
        user_data, chat_id = await get_chat_data(message)
        model = 'gpt-3.5-turbo' if not user_data.get('gpt-4', config.useGPT4) else 'gpt-4'

        results = await query_index(chat_id, text,[file_path.name],model)
        if results:
            await message.reply( f"Found results: {results}")
        else:
            await message.reply( "No results found.")
    except:
        traceback.print_exc()
        await message.reply(traceback.format_exc())
