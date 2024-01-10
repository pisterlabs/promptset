import discord
import os
from dotenv import load_dotenv
from tastytrade_sdk import Tastytrade
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import math
import pandas as pd
from openai import OpenAI
from datetime import datetime, timedelta
import pytz
import logging
from dateutil.easter import easter
from dateutil.relativedelta import MO, TH, relativedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
scheduler = AsyncIOScheduler()

# Initialize the Tastytrade client
tasty = Tastytrade()
ACCOUNT_NUMBER = os.getenv('ACCOUNT_NUMBER')
LOGIN = os.getenv('LOGIN')
PASSWORD = os.getenv('PASSWORD')
TOKEN = os.getenv('NASSAU_GPT_TOKEN')
CHANNEL_ID = int(os.getenv('PORTFOLIO_CHANNEL'))
nassau_gpt = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'gpt-4-1106-preview'

# Define intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True


# Initialize the Discord client with intents
client = discord.Client(intents=intents)


def get_good_friday(year):
    return easter(year) - timedelta(days=2)


def adjust_for_weekend(holiday):
    if holiday.weekday() == 5:  # Saturday
        return holiday - timedelta(days=1)
    elif holiday.weekday() == 6:  # Sunday
        return holiday + timedelta(days=1)
    return holiday


def get_holidays(year):
    return {
        datetime(year, 1, 1).date(): "New Year's Day",
        datetime(year - 1, 12, 31).date(): "New Year's Day Observed",
        datetime(year, 7, 4).date(): 'Independence Day',
        datetime(year, 12, 25).date(): 'Christmas Day',
        (
            datetime(year, 1, 1) + relativedelta(weekday=MO(3))
        ).date(): 'Martin Luther King Jr. Day',
        (
            datetime(year, 2, 1) + relativedelta(weekday=MO(3))
        ).date(): "Washington's Birthday",
        (datetime(year, 5, 31) - relativedelta(weekday=MO(-1))).date(): 'Memorial Day',
        (datetime(year, 9, 1) + relativedelta(weekday=MO(1))).date(): 'Labor Day',
        (datetime(year, 11, 1) + relativedelta(weekday=TH(4))).date(): 'Thanksgiving',
        get_good_friday(year): 'Good Friday',
    }


def stock_market_holiday(date):
    year = date.year
    holidays = get_holidays(year)

    # Adjust for observed holidays on weekends
    adjusted_holidays = {
        adjust_for_weekend(holiday_date): name
        for holiday_date, name in holidays.items()
    }
    return date in adjusted_holidays


def get_holiday_name(date):
    year = date.year
    holidays = get_holidays(year)

    # Adjust for observed holidays on weekends
    adjusted_holidays = {
        adjust_for_weekend(holiday_date): name
        for holiday_date, name in holidays.items()
    }
    return adjusted_holidays.get(date)


def generate_prelude():
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    # today = mock_today_monday_morning.date()
    day_of_week = today.weekday()
    current_time = datetime.now().time()
    morning_cutoff = datetime.strptime('09:30', '%H:%M').time()

    monday_message = "It's Monday! Get the Discord server hyped to kick off the trading week! Time to kickstart a powerful week. "
    friday_message = (
        "It's Friday, so send everyone in the Discord server a hearty weekend goodbye!"
    )

    if stock_market_holiday(today):
        holiday = get_holiday_name(today)
        prompt = (
            f'Today is {holiday}. Write a concise message to the Discord server of the elite student-run investment club Nassau Street Capital at Princeton '
            'wishing them a wonderful break! Sign warmly off as NassauBot.'
        )
    elif current_time < morning_cutoff:
        prompt = (
            (monday_message if day_of_week == 0 else '')
            + 'Write a concise, deadpan market opening statement in the style of a deadpan Wall Street VP with '
            'a WallStreetBets / Wolf of Wall Street twist for an elite, student-run investment club Nassau Street Capital. '
            'Should be no more than a few sentences, but the deadpan part is what makes it so damn funny. '
            "Don't sound like a thesaurus. This opener should have a dry wit, delivering a sharp, humorously understated rallying cry "
            "for the day's trading. Keep it brief, clever, with just a hint of intelligent arrogance."
        )
    else:
        prompt = (
            "Write a succinct, deadpan closing statement, combining corporate finance's seriousness with WallStreetBets flair, "
            'for the elite, student-run investment club Nassau Street Capital at Princeton. This end-of-day remark should convey a mix of dry satisfaction and '
            "subtle humor. It should be concise, reflecting the day's (hopefully) successful trading in a witty, understated way. "
            + (
                "Always end with some sort of variant (can be unchanged as well, swearing also optional) of 'Fuck you and I'll see you tomorrow'."
                if day_of_week != 4
                else friday_message
            )
        )
    try:
        response = nassau_gpt.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': prompt,
                }
            ],
            temperature=1.01,
            frequency_penalty=0.9,
            presence_penalty=0.9,
        )
        prelude_statement = response.choices[0].message.content.strip()
        return prelude_statement

    except Exception as ex:
        logger.error('Error generating prelude statement: %s', ex)
        return "Forgot to draft up a dank statement for today. Anyway... here's our club portfolio value:"


def midday_update():
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    if stock_market_holiday(today):
        return
    midday_prompt = (
        'Craft a midday market update in the style of a seasoned Wall Street trader for the elite Nassau Street Capital club. '
        'This update should include a mix of savvy market insights and a touch of humor, presented in a concise, deadpan manner. '
        "The message should offer a brief snapshot of the day's market movements so far, highlighting any significant trends or events."
    )
    try:
        response = nassau_gpt.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': midday_prompt,
                }
            ],
            temperature=1.025,
            frequency_penalty=0.9,
            presence_penalty=0.9,
        )
        midday_statement = response.choices[0].message.content.strip()
        return midday_statement

    except Exception as ex:
        logger.error('Error generating midday statement: %s', ex)
        return "Seems like the market took a lunch break, and so did my messaging skills. But here's where we stand at midday:"


def power_hour_update():
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    if stock_market_holiday(today):
        return
    power_hour_prompt = (
        'Compose an energizing power hour statement for the Nassau Street Capital club, capturing the thrill of the last trading hour. '
        'This message should be charged with enthusiasm and a hint of audacity, typical of a final-hour trading rush. '
        "Encapsulate the urgency and potential of the market's closing hour in a sharp, witty remark."
    )
    try:
        response = nassau_gpt.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': power_hour_prompt,
                }
            ],
            temperature=1.025,
            frequency_penalty=0.9,
            presence_penalty=0.9,
        )
        power_hour_statement = response.choices[0].message.content.strip()
        return power_hour_statement

    except Exception as ex:
        logger.error('Error generating power hour statement: %s', ex)
        return "As the market's final bell looms, my ability to craft messages seems to be fading. But, here's the crunch-time update:"


async def report_balance():
    prelude = generate_prelude()
    channel = client.get_channel(CHANNEL_ID)
    tasty.login(login=LOGIN, password=PASSWORD)
    account = tasty.api.get(f'/accounts/{ACCOUNT_NUMBER}/balances')
    balance = account['data']['cash-balance']
    formatted_balance = math.ceil(float(balance) * 100) / 100
    formatted_balance_with_currency = '${:,.2f}'.format(formatted_balance)
    await channel.send(
        f'{prelude}\n**Portfolio Balance:** `{formatted_balance_with_currency}`\n'
    )


async def report_positions():
    positions = tasty.api.get(f'/accounts/{ACCOUNT_NUMBER}/positions')
    positions_df = pd.DataFrame(positions['data'])
    print(positions_df)


@client.event
async def on_ready():
    est = pytz.timezone('US/Eastern')
    # Morning report
    scheduler.add_job(scheduled_report, 'cron', hour=9, minute=25, timezone=est)
    # Midday update
    scheduler.add_job(midday_update, 'cron', hour=12, minute=30, timezone=est)
    # Power hour update
    scheduler.add_job(power_hour_update, 'cron', hour=15, minute=0, timezone=est)
    # End of day report
    scheduler.add_job(scheduled_report, 'cron', hour=15, minute=59, timezone=est)

    scheduler.start()


async def scheduled_report():
    await report_balance()
    # await report_positions()


client.run(TOKEN)
