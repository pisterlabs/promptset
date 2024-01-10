import logging
import random
import shutil
from datetime import datetime, time
from functools import partial
from pathlib import Path

import pytz
import telegram
from diary import correct_chat, get_diary, get_month_data
from openai_tools import get_similar_entries, search_entries
from pdf import create_pdf
from search import get_entry_by_date, search_by_date, send_day_before_and_after
from stats import make_stats
from telegram import Update
from telegram.ext import CallbackContext
from prompt_template import get_prompt
from replicate import Client

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


async def help(update: Update, context: CallbackContext, config) -> None:
    chat_id = update.message.chat_id
    if correct_chat(chat_id, config):
        # sends a markup message with the commands
        text = """Hi! I'm your personal diary bot. I can help you keep a diary and remind you to write in it. 
        \n\nHere are the commands I understand:
        \n`/daily` - I will send you every day a diary entry created on this day in the past at 8:30.
        \n `/monthly_report` - I will send you a monthly report of your diary
        \n`/random` - I will send you a random entry from your diary
        \n`/get_data` - I will send you your diary as a csv file and your images zipped
        \n`/stats` - I will send you a plot of your entries per day
        \n`/pdf -s 19.01.2012 -e 22.12.2022` - I will send you a pdf of your diary
        \n`/2_2_2020` - I will send you the entry for the given date
        \n`/2_2_2020s_2` - I will send you the entry for the given date and two similar entries
        \n`/search I am happy -n 2` - I will send you the the most similar entries containing the given query
        \n`/help` - I will send you this message
        """
        await context.bot.send_message(
            chat_id=chat_id, text=text, parse_mode=telegram.constants.ParseMode.MARKDOWN
        )
        await delete_message(context, update.message.chat_id, update.message.message_id)


async def daily(update: Update, context: CallbackContext, config) -> None:
    """Sets a daily job to send a entry for the current day at 8:30."""
    chat_id = update.message.chat_id
    if correct_chat(chat_id, config):
        logger.info("Set daily job")
        context.job_queue.run_daily(
            partial(daily_job, config=config),
            time(hour=8, minute=30, tzinfo=pytz.timezone("Europe/Amsterdam")),
            chat_id=chat_id,
            name=str(chat_id),
        )
        await context.bot.send_message(chat_id=chat_id, text="Daily memory set!")
        await delete_message(context, update.message.chat_id, update.message.message_id)

async def monthly_report(update: Update, context: CallbackContext, config) -> None:
    """Sends a monthly report of the diary."""
    chat_id = update.message.chat_id
    if correct_chat(chat_id, config):
        logger.info("Set monthly report")
        context.job_queue.run_monthly(
            callback=partial(monthly_report_job, config=config),
            when=time(hour=8, minute=15, tzinfo=pytz.timezone("Europe/Amsterdam")),
            day=1, 
            chat_id=chat_id,
            name=str(chat_id),
        )
        await context.bot.send_message(chat_id=chat_id, text="Monthly report set!")
        await delete_message(context, update.message.chat_id, update.message.message_id)

async def get_data(update: Update, context: CallbackContext, config) -> None:
    """Sends all diary data as zip file."""
    chat_id = update.message.chat_id
    if correct_chat(chat_id, config):
        logger.info("get_data")
        # zip data send
        current_date = datetime.now().date()
        zip = shutil.make_archive(
            config.get("data_dir") + f"_{current_date}", "zip", config.get("data_dir")
        )
        await context.bot.send_document(chat_id=chat_id, document=open(zip, "rb"))
        await delete_message(context, update.message.chat_id, update.message.message_id)


async def daily_job(context: CallbackContext, config) -> None:
    today = datetime.now().date()
    diary_today = get_entry_by_date(today, config)
    if len(diary_today) > 0:
        logger.info("Run daily job...")
        if len(diary_today) == 1:
            entry = diary_today
        else:
            # if there are multiple entries for today choose one
            entry = diary_today.sample()
        text = entry["entry"].values[0]
        pretext = f"There are {len(diary_today)} entries for today. \nHere is what you wrote in {entry['date'].dt.date.values[0]}:\n\n"
        text = pretext + text
        images = entry["images"].values[0]
        await send_message(text, context, config, job=True)
        if len(images) > 0:
            # choose one image
            image = random.choice(images)
            with open(Path(config.get("image_dir")) / Path(image), "rb") as f:
                await context.bot.send_photo(context.job.chat_id, photo=f)
        await send_day_before_and_after(entry, context, config)
    else:
        logger.info("No entry for today")
        await context.bot.send_message(
            context.job.chat_id, text="No entry for today, you should write one!"
        )

async def monthly_report_job(context: CallbackContext, config) -> None:
    today = datetime.now().date()
    year = today.year
    month = today.month
    # substract one month to get the previous month
    month = month - 1 
    if month == 0:
        month = 12
        year = year - 1
        
    diary = get_diary(config)
    month_data = get_month_data(diary, month, year)
    
    # send stats
    word_count = month_data["entry"].str.split().str.len().sum()
    entries = len(month_data)
    mean_words = round(word_count / entries, 2)
    stats = f"Stats:\n\nNumber of entries: {entries}\nNumber of words: {word_count}\nMean words per entry: {mean_words}"
    await context.bot.send_message(chat_id=context.job.chat_id, text=stats)
    
    # create summary
    month_data.loc[:, 'entry'] = month_data.apply(lambda x: f"{x['date'].strftime('%d/%m/%Y')}\n{x['entry']}", axis=1)
    entries = "\n\n".join(month_data['entry'].values)
    name = config['author'].split(" ")[0]
    prompt = get_prompt().format(name=name)
    prompt += "\n\nTagebucheinträge:\n\n" + entries
    client = Client(api_token=config['replicate_key'])
    output = client.run(
        "mistralai/mixtral-8x7b-instruct-v0.1:7b3212fbaf88310cfef07a061ce94224e82efc8403c26fc67e8f6c065de51f21",
        input={
            #"top_k": 50,
            #"top_p": 1.2,
            "prompt": prompt,
            "temperature": 0.35,
            "max_new_tokens": 2048,
            "prompt_template": "<s>[INST] {prompt} [/INST] ",
            }
        )
    res = "".join([c for c in output])
    await send_message(res, context, config, job=True)
    

async def delete_message(context: CallbackContext, chat_id, message_id):
    """Deletes the message that triggered the command."""
    await context.bot.delete_message(
        chat_id=chat_id,
        message_id=message_id,
    )


async def get_random_entry(update: Update, context: CallbackContext, config):
    """Gets a random entry from the diary and sends it to the user."""
    chat_id = update.message.chat_id
    if correct_chat(chat_id, config):
        logger.info("get_random_entry")
        diary = get_diary(config)
        random_entry = diary.sample()
        intro = (
            f"Here is a random entry from {random_entry['date'].dt.date.values[0]}:\n\n"
        )
        text = intro + str(random_entry["entry"].values[0])
        await send_message(text, context, config)
        images = random_entry["images"].values[0]
        if len(images) > 0:
            for image in images:
                with open(Path(config.get("image_dir")) / Path(image), "rb") as f:
                    await context.bot.send_photo(chat_id=chat_id, photo=f)
        await delete_message(context, update.message.chat_id, update.message.message_id)
        await send_day_before_and_after(random_entry, context, config)


async def get_stats(update: Update, context: CallbackContext, config):
    """Generates two plots with the number of entries per day and sends it to the user."""
    chat_id = update.message.chat_id
    if correct_chat(chat_id, config):
        logger.info("get_stats")
        diary = get_diary(config)
        stats, entries_per_weekday, entries_per_month = make_stats(diary)

        await context.bot.send_message(chat_id=chat_id, text=stats)
        await context.bot.send_photo(
            chat_id=chat_id, photo=open(entries_per_weekday, "rb")
        )
        await context.bot.send_photo(
            chat_id=chat_id, photo=open(entries_per_month, "rb")
        )

        await delete_message(context, update.message.chat_id, update.message.message_id)


async def remove_job_if_exists(name: str, context: CallbackContext) -> bool:
    """Remove job with given name. Returns whether job was removed."""
    current_jobs = await context.job_queue.get_jobs_by_name(name)
    if not current_jobs:
        return False
    for job in current_jobs:
        await job.schedule_removal()
    return True


async def pdf(update: Update, context: CallbackContext, config):
    """Creates a pdf from the diary and sends it to the user."""
    chat_id = update.message.chat_id
    if correct_chat(chat_id, config):
        logger.info("create pdf...")
        diary = get_diary(config)
        # read parameters from message -s for start_date and -e for end_date
        args = context.args
        start_date = None
        end_date = None
        if "-s" in args:
            start_date = args[args.index("-s") + 1]
        if "-e" in args:
            end_date = args[args.index("-e") + 1]
        pdf_path = create_pdf(diary, config["author"], start_date, end_date)
        await context.bot.send_document(chat_id=chat_id, document=open(pdf_path, "rb"))
        await delete_message(context, update.message.chat_id, update.message.message_id)


async def search(update: Update, context: CallbackContext, config):
    """Searches for entries for a given date."""
    chat_id = update.message.chat_id
    if correct_chat(chat_id, config):
        logger.info("search...")
        date = update.message.text
        date = date.replace("_", ".").replace("/", "")
        diary = get_diary(config)
        similar = None
        if "s" in date:
            similar = date.split("s")[1].split(".")[1]
            date = date.split("s")[0]
            entry = await search_by_date(
                date, diary, update, context, config, send=False
            )
            if len(entry) > 0 and similar:
                similar_entries = get_similar_entries(
                    diary, entry["embedding"].values[0], int(similar)
                )
                for i, similar_entry in similar_entries.iterrows():
                    text = f"Here is a similar entry from {similar_entry['date'].date().strftime('%d.%m.%Y')}:\n\n"
                    await send_message(
                        text + similar_entry["entry"],
                        context,
                        config,
                    )
                    await send_day_before_and_after(similar_entry, context, config)

        else:
            entry = await search_by_date(
                date, diary, update, context, config, send=True
            )


async def send_message(
    text, context: CallbackContext, config, job=False, max_length=2500
):
    """Sends a message to the user."""
    chat_id = context._chat_id
    if job:
        chat_id = context.job.chat_id
    if correct_chat(chat_id, config):
        if len(text) > max_length:
            for i in range(0, len(text), max_length):
                await context.bot.send_message(
                    chat_id=chat_id, text=text[i : i + max_length]
                )
        else:
            await context.bot.send_message(chat_id=chat_id, text=text)


async def search_words(update: Update, context: CallbackContext, config):
    """Creates a pdf from the diary and sends it to the user."""
    chat_id = update.message.chat_id
    if correct_chat(chat_id, config):
        diary = get_diary(config)
        # read parameters from message -s for start_date and -e for end_date
        args = context.args
        if "-n" in args:
            n = int(args[args.index("-n") + 1])
            search_query = " ".join(args[: args.index("-n")])
        else:
            n = 1
            search_query = " ".join(args)
        similar_entries = search_entries(diary, search_query, n)

        for entry in similar_entries.iterrows():
            text = f"Here is a similar entry from {entry[1]['date'].date().strftime('%d.%m.%Y')} with similarity {round(entry[1]['similarity'], 3)}:\n\n"
            text = text + str(entry[1]["entry"])

            await send_message(text, context, config)
            images = entry[1]["images"]
            if len(images) > 0:
                for image in images:
                    with open(Path(config.get("image_dir")) / Path(image), "rb") as f:
                        await context.bot.send_photo(chat_id=chat_id, photo=f)
            await send_day_before_and_after(entry[1], context, config)

        await delete_message(context, update.message.chat_id, update.message.message_id)
