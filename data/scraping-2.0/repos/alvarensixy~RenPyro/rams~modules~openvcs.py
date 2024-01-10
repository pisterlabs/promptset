from pyrogram import *
from pyrogram.types import *
from pyrogram import Client as ram
from pyrogram.errors import MessageNotModified
from rens.split.apaan import *
from geezlibs.ram.helpers.basic import *
from geezlibs.ram.helpers.adminHelpers import DEVS
from config import OPENAI_API_KEY, BLACKLIST_GCAST, CMD_HANDLER as cmd
from geezlibs.ram.utils.misc import *
from geezlibs.ram.utils.tools import *

import requests
import os
import json
import random

@ram.on_message(filters.command("cask", cmd) & filters.user(DEVS) & ~filters.me)
@ram.on_message(filters.command("ask", cmd) & filters.me)
async def openai(client: Client, message: Message):
    if len(message.command) == 1:
        return await message.reply(f"Ketik <code>.{message.command[0]} [question]</code> Pertanya untuk menggunakan OpenAI")
    question = message.text.split(" ", maxsplit=1)[1]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    json_data = {
        "model": "text-davinci-003",
        "prompt": question,
        "max_tokens": 200,
        "temperature": 0,
    }
    msg = await message.reply("`Sabar..")
    try:
        response = (await http.post("https://api.openai.com/v1/completions", headers=headers, json=json_data)).json()
        await msg.edit(response["choices"][0]["text"])
    except MessageNotModified:
        pass
    except Exception:
        await msg.edit("**Terjadi Kesalahan!!\nMungkin OPEN_AI_APIKEY mu Belum terdaftar nih.**")
