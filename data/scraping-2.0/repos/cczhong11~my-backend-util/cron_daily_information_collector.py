import random
from typing import List

from bs4 import BeautifulSoup
from DataFetcher.JiuDianArticleFetcher import JiudianArticleFetcher
from DataFetcher.TTSFetcher import TTSFetcher
from DataFetcher.AccuweatherDataFetcher import AccuweatherDataFetcher
from DataFetcher.GMailDataFetcher import GMailDataFetcher, NewsLetterType
from DataFetcher.GoogleCalendarDataFetcher import GoogleCalendarDataFetcher
from DataFetcher.RSSDataFetcher import RSSDataFetcher
from DataReader.PersonalBackendReader import PersonalBackendReader
from DataFetcher.ZhihuArticleFetcher import ZhihuArticleFetcher
from DataWriter.AWSS3DataWriter import AWSS3DataWriter
from DataWriter.OpenAIDataWriter import OpenAIDataWriter
from constant import PATH
import time_util
import dataclasses
import re
from web_util import read_json_file, parse_curl
import os
import datetime
from gcsa.event import Event
import os
import log_util
from util import get_tts_path

logger = log_util.get_logger("cron_daily_information_collector")

DEBUG = False
# if mac, debug = True
if os.uname().sysname == "Darwin":
    DEBUG = True


@dataclasses.dataclass
class BookWisdom:
    book_name: str
    wisdoms: list


@dataclasses.dataclass
class DailyInformation:
    events: list
    wsj_news: str
    weather: str
    hacker_news: list
    book_text: BookWisdom
    meitou: list
    tech_summary: str

    def __str__(self):
        events = "\n".join(
            [
                f"从{e.start.hour}:{e.start.minute}到{e.end.hour}:{e.end.minute} {e.summary} {e.description or ''}"
                for e in self.events or []
                if isinstance(e.start, datetime.datetime)
                and e.summary != "ME"
                and e.summary != "wmtd"
            ]
        )
        wisdom = "\n".join(self.book_text.wisdoms)
        meitou = "\n".join(self.meitou)
        return f"""
        今天的日历: {events}
        今天的新闻: {self.wsj_news}
        今天的天气: {self.weather}
        hacker news: {self.hacker_news}
        读过的书: {self.book_text.book_name}
        {wisdom}
        投资新闻: {meitou}
        """


today = time_util.str_time(time_util.get_current_date(), "%Y/%m/%d")
today_str = time_util.str_time(time_util.get_current_date(), "%Y-%m-%d")
yesterday = time_util.str_time(time_util.get_yesterday(), "%Y/%m/%d")
tomorrow = time_util.str_time(time_util.get_next_day(), "%Y/%m/%d")
api = read_json_file(f"{PATH}/key.json")
life_calendar = api["life_calendar"]
main_calendar = api["main_calendar"]
personal_backend_url = api["personal_backend_url"]
openai = OpenAIDataWriter(api["openai"])


def clean_wsj(content):
    cleaned_content = re.sub(r"â\x80\x8c\s*", "", content)
    lines = cleaned_content.split("\n")

    # Keeping lines that contain alphanumeric characters
    readable_lines = [line for line in lines if re.search(r"\w", line)]

    new_lines = []
    for line in readable_lines:
        if "Email us your comments, which we may edit" in line:
            break
        new_lines.append(line)
    readable_content = "\n".join(new_lines)
    return readable_content


def run_calendar() -> List[Event]:
    cal = GoogleCalendarDataFetcher(
        secret_file=None, credentials_file=f"{PATH}/cookie/gmailcredentials.json"
    )
    cal.load_cookie()
    events = cal.get_data(
        life_calendar, time_util.get_current_date(), time_util.get_next_day()
    )
    result = []
    for e in events:
        if e.summary != "ME" and e.summary != "wmtd":
            result.append(e)
    main_events = cal.get_data(
        "primary", time_util.get_current_date(), time_util.get_next_day()
    )
    for e in main_events:
        if (
            e.summary != "ME"
            and e.summary != "wmtd"
            and "°C" not in e.summary
            and "logged" not in e.summary
        ):
            result.append(e)
    return result


def run_wsj():
    gmail = GMailDataFetcher(
        f"{PATH}/cookie/gmail.json", f"{PATH}/cookie/gmailcredentials.json"
    )
    if not gmail.health_check():
        raise Exception("gmail health check failed")
    gmail.reset_query()
    gmail.set_sender(NewsLetterType.WSJ)
    gmail.set_time(today, tomorrow)
    gmail.add_query(" The 10-Point ")
    tmp = gmail.get_data()
    rs = gmail.analyze(NewsLetterType.WSJ, tmp)
    wsj = rs[0]["content"]
    wsj = clean_wsj(wsj)

    wsj_summary = openai.summary_data(
        "这是一段10条新闻。先说新闻，最后说说在那些方面的股票可能会受到上述新闻的影响。", wsj[:6000], use_16k_model=False
    )
    return wsj_summary


def clean_tech(content):
    # find html tag
    soup = BeautifulSoup(content, "html.parser")
    text_content = soup.text
    text_content = re.sub(r"â\x80\x8c\s*", "", text_content)
    lines = text_content.split("\n")
    # skip first TechCrunch Newsletter
    start_index = 1
    current_index = 1
    for line in lines[1:]:
        if "TechCrunch Newsletter" in line:
            start_index = current_index + 1
            break
        current_index += 1
    return "\n".join(lines[start_index:])


def run_techcrunch():
    gmail = GMailDataFetcher(
        f"{PATH}/cookie/gmail.json", f"{PATH}/cookie/gmailcredentials.json"
    )
    if not gmail.health_check():
        raise Exception("gmail health check failed")
    gmail.reset_query()
    gmail.set_sender(NewsLetterType.TECHCRUNCH)
    gmail.set_time(today, tomorrow)
    tmp = gmail.get_data()
    rs = gmail.analyze(NewsLetterType.TECHCRUNCH, tmp)
    tech = rs[0]["content"]

    tech = clean_tech(tech)

    tech_summary = openai.summary_data(
        "这是一些科技新闻，用中文总结一下。", tech[:6000], use_16k_model=False
    )
    return tech_summary


def run_weather():
    weather = AccuweatherDataFetcher(api["accuweather"], "sunnyvale")
    return weather.get_data()


def run_hacker_news():
    rss = RSSDataFetcher("https://hnrss.org/newest?points=500")
    news = rss.fetch(time_util.get_yesterday())
    return [n["title"] for n in news]


def run_book_wisdom():
    books = PersonalBackendReader(personal_backend_url)
    # choose a book
    random_book = random.choice(books.get_list("kindle")["data"])
    book_name = random_book["name"].split(".")[0]
    book_text = books.get_data("kindle", book_name)
    random_five = random.sample(book_text["data"][0]["data"].get("content"), 5)
    return BookWisdom(book_name, random_five)


def run_meitou():
    jd = JiudianArticleFetcher()
    if not jd.health_check():
        print("jd health check failed")
        return []
    return [
        f"视频标题:{d['title']} 发布频道:{d['yt_chan_title']} 内容总结:{' '.join(d['extracted_texts'])}"
        for d in jd.get_data()
    ]


def run():
    logger.info("start cron_daily_information_collector")
    collect_f = {
        "events": run_calendar,
        "wsj_summary": run_wsj,
        "weather": run_weather,
        "hacker_news": run_hacker_news,
        "book_text": run_book_wisdom,
        "meitou": run_meitou,
        "tech_summary": run_techcrunch,
    }
    result = {}
    for k, v in collect_f.items():
        try:
            result[k] = v()
        except Exception as e:
            logger.error(f"{k} failed, error: {e}")
            if k == "events" or k == "hacker_news" or k == "meitou":
                result[k] = []
            else:
                result[k] = ""
    daily_info = DailyInformation(
        result["events"],
        result["wsj_summary"],
        result["weather"],
        result["hacker_news"],
        result["book_text"],
        result["meitou"],
        result["tech_summary"],
    )
    tts_path = get_tts_path()
    tts = TTSFetcher(f"{PATH}/cookie/tts-google.json", tts_path)
    s3 = AWSS3DataWriter("rss-ztc")
    result = str(daily_info)
    if DEBUG:
        print(result)
        return

    with open(f"{tts_path}/{today_str}.txt", "w") as f:
        f.write(result)
    tts.get_tts_from_text(result, today_str)
    s3.write_data(
        "daily",
        os.path.join(f"{tts_path}", f"{tts.current_title()}.mp3"),
    )


if __name__ == "__main__":
    run()
