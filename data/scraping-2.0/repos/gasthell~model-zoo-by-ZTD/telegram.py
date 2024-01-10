from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.types import InputFile
from aiogram.utils import executor
import json
import requests
import io
import base64
from PIL import Image

url = "http://127.0.0.1:7860"

TOKEN = 'TOKEN'

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

template = """You are ChatOSZ, a language model created by ZTD. You speak only on english. You help with writing research. Your main source of information is https://osdr.nasa.gov/. And you could search from another source if you couldnt find information
Human: {human_input}
ChatOSZ:"""

prompt = PromptTemplate(input_variables=["human_input"], template=template)

cllama_chain = LLMChain(
    llm="OSZ.bin",
    prompt=prompt,
    verbose=False,
    memory=ConversationBufferWindowMemory(k=2),
    llm_kwargs={"max_length": 4096}
)

def qchat(input1):
    loader = WebBaseLoader("https://scholar.google.com/"+input1)
    docs = loader.load()
    chain = load_summarize_chain(cllama_chain, chain_type="stuff")
    chain.run(docs)
    output = cllama_chain.predict(input1)
    return(output)

def cchat(input1):
    loader = WebBaseLoader("https://scholar.google.com/"+input1)
    docs = loader.load()
    chain = load_summarize_chain(cllama_chain, chain_type="stuff")
    chain.run(docs)
    output = cllama_chain.predict("Check this text: " + input1)
    return(output)

def dreschat(input1):
    loader = WebBaseLoader("https://scholar.google.com/"+input1)
    docs = loader.load()
    chain = load_summarize_chain(cllama_chain, chain_type="stuff")
    chain.run(docs)
    output = cllama_chain.predict("Design and correct this research(maximum 2000 symbols). At start write tags for this text: " + input1)
    return(output)

def subjectchat(input1):
    loader = WebBaseLoader("https://scholar.google.com/"+input1)
    docs = loader.load()
    chain = load_summarize_chain(cllama_chain, chain_type="stuff")
    chain.run(docs)
    output = cllama_chain.predict("Write near which 1 subject and nothing else the research is taking place: " + input1)
    return(output)

@dp.message_handler(commands=['start', 'help'])
async def process_start_command(message: types.Message):
    await message.reply("Hello! I am ChatOSZ, an artificial intelligence trained by ZTD for writing research.\n/ask - for ask to bot\n/check - to check text\n/generate - to make image\n/dres - to design your research")

@dp.message_handler(commands=['ask'])
async def process_ask_command(message: types.Message):
    await message.reply(qchat(message.text))

@dp.message_handler(commands=['dres'])
async def process_dres_command(message: types.Message):
    subject = subjectchat(message.text)
    payload = {
        "prompt": "black "+subject+", white background, an elegant, timeless, youthful logo",
        "negative_prompt": "text, realistic",
        "sampler_index": "DPM++ 2M Karras",
        "batch_size": 5,
        "cfg_scale": 8,
        "height": 512,
        "width": 512,
        "steps": 50,
        "seed": -1,
    }
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    r = response.json()
    for i in range(5):
        image = Image.open(io.BytesIO(base64.b64decode(r['images'][i])))
        image.save('output.png')
        photo = open('output.png', 'rb')
        await bot.send_photo(message.chat.id, photo)
    await message.reply(dreschat(message.text))


@dp.message_handler(commands=['check'])
async def process_check_command(message: types.Message):
    await message.reply(cchat(message.text))

@dp.message_handler(commands=['generate'])
async def process_generate_command(message: types.Message):
    payload = {
        "prompt": "black "+message.text+", white background, an elegant, timeless, youthful logo",
        "negative_prompt": "text, realistic",
        "sampler_index": "DPM++ 2M Karras",
        "batch_size": 5,
        "cfg_scale": 8,
        "height": 512,
        "width": 512,
        "steps": 50,
        "seed": -1,
    }
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    r = response.json()
    for i in range(5):
        image = Image.open(io.BytesIO(base64.b64decode(r['images'][i])))
        image.save('output.png')
        photo = open('output.png', 'rb')
        await bot.send_photo(message.chat.id, photo)

if __name__ == '__main__':
    executor.start_polling(dp)