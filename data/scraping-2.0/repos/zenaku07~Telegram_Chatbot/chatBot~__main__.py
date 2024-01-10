import asyncio

import os

import openai

from pyrogram import filters, idle

import sqlite3

#-------------------------------------#

from rich.console import Console

from rich.table import Table

from rich.panel import Panel

from rich.box import SIMPLE_HEAVY

#-------------------------------------#

from . import Soham

from setup import guu

conn = sqlite3.connect('feedback.db')

c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS Feedback

             (id INTEGER PRIMARY KEY AUTOINCREMENT,

              user_id INTEGER,

              username TEXT,

              feedback TEXT)''')

conn.commit()

async def ai(query):

    openai.api_key = guu.openai_api_key

    completion = openai.Completion.create(engine=guu.model, prompt=query, max_tokens=guu.mxtoken, n=1, stop=None,temperature=0.7)

    result = completion.choices[0].text

    return result

@Soham.on_message(filters.command("feedback") & ~filters.group)

async def feedback_handler(bot, msg):

    feedback = msg.text.replace("/feedback ", "")

    user_id = msg.from_user.id

    username = msg.from_user.username

    c.execute("INSERT INTO Feedback (user_id, username, feedback) VALUES (?, ?, ?)", (user_id, username, feedback))

    conn.commit()

    chat = await bot.get_chat("cosmic98")

    chat_id = chat.id

    await bot.send_message(

        chat_id=chat_id,

        text=f"New feedback from @{username} (user ID: {user_id}):\n{feedback}"

    )

    await bot.send_message(

        chat_id=user_id,

        text="Thank you for your feedback! We appreciate your response 🙂."

    )

@Soham.on_message(filters.command("start") & ~filters.group)

async def main(bot,msg):

    newbie = msg.from_user.id

    await bot.send_message(newbie, "" '👋 Hello ''!\n\n'

                    'My name is Hinata Hyuga. \n'

                    'And I am a telegram AI based chat-bot \n\n'

                    'Belongs to OpenAIs GPT-3 family \n'

                    'Im here to help answer any questions you may have about a variety of topics.\n'

                    'Feel free to ask me anything! ☺️\n\n'

                    'MADE BY : @Sync_0  \n'

                    'Git-Hub Profile : https://github.com/blue0777\n'

                    'Git-Hub Repository : http://github.com/blue0777/TG-ChatGPT-Bot\n\n'

                    'We would love to hear your thoughts on this Telegram chat bot. \n'

                    'Your feedback will help us enhance our service.\n'

                    'Please leave a brief feedback  below you can send feedback just typing \n /feedback "your feedback 🙂\n'

                    'we are appreciate your feedback . Thank you!"'

                    )

    DEL = await msg.reply(f"Typing🤔.......")

    await asyncio.sleep(3)

    await DEL.delete(10)

@Soham.on_message(filters.text & ~filters.group)

async def main(bot, msg):

    newbie = msg.from_user.id

    ques = msg.text

    print(ques)

    guu = await ai(ques)

    await asyncio.sleep(3)

    print(guu)

    test = f"{guu}"

    await asyncio.sleep(1)

    await bot.send_message(newbie,test)

if name == "main":

  #-------------------------------------------------------------------#

    Soham.start()

    console = Console()

    console.print("\n" * 2)

    console.print(Panel("Bot Started", title="[bold blue]BOT", border_style="bold blue", padding=(1, 2)))

    console.print("\n" * 2)

    console.print(Panel(f"{Soham.get_me().username}", title="[bold blue_violet]USERNAME", border_style="bold blue_violet", padding=(1, 2)))

    console.print("\n" * 2)

    table = Table(title="MODULES: ", show_header=False, box=SIMPLE_HEAVY)

    table.add_row("1 : ChatBot Module")

    table.add_row("2 : OpenAiApi Module")

    console.print(Panel(table, title="[bold green]STATUS", border_style="bold green", padding=(1, 2)))

    console.print("\n" * 2)

    console.print("\n" * 2)

    console.rule("Other Logs", style="bold magenta")

    console.print("\n" * 2)

    idle()




    

