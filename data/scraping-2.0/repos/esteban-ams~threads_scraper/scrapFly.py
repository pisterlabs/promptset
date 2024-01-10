import json
import jmespath
import datetime
import pandas as pd
import time

from typing import Dict
from parsel import Selector
from nested_lookup import nested_lookup
from playwright.sync_api import sync_playwright

# from compa_project.OpenAiSentiment import openaiInit, obtener_sentimiento

# openaiInit()

threads_list = (
    # (1, 2) #  thread url, sentimiento, entre otros
    # ("","","",""),

)

def parse_thread(data: Dict) -> Dict:
    """Parse Twitter tweet JSON dataset for the most important fields"""
    result = jmespath.search(
        """{
        text: post.caption.text,
        published_on: post.taken_at,
        id: post.id,
        pk: post.pk,
        code: post.code,
        username: post.user.username,
        user_pic: post.user.profile_pic_url,
        user_verified: post.user.is_verified,
        user_pk: post.user.pk,
        user_id: post.user.id,
        has_audio: post.has_audio,
        reply_count: view_replies_cta_string,
        like_count: post.like_count,
        images: post.carousel_media[].image_versions2.candidates[1].url,
        image_count: post.carousel_media_count,
        videos: post.video_versions[].url
    }""",
        data,
    )
    result["videos"] = list(set(result["videos"] or []))
    if result["reply_count"]:
        result["reply_count"] = int(result["reply_count"].split(" ")[0])
    result[
        "url"
    ] = f"https://www.threads.net/@{result['username']}/post/{result['code']}"
    return result


def scrape_thread(url: str) -> dict:
    """Scrape Threads post and replies from a given URL"""
    with sync_playwright() as pw:
        # start Playwright browser
        browser = pw.chromium.launch()
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # go to url and wait for the page to load
        page.goto(url)
        # wait for page to finish loading
        page.wait_for_selector("[data-pressable-container=true]")
        # find all hidden datasets
        selector = Selector(page.content())
        hidden_datasets = selector.css('script[type="application/json"][data-sjs]::text').getall()
        # find datasets that contain threads data
        for hidden_dataset in hidden_datasets:
            # skip loading datasets that clearly don't contain threads data
            if '"ScheduledServerJS"' not in hidden_dataset:
                continue
            if "thread_items" not in hidden_dataset:
                continue
            data = json.loads(hidden_dataset)
            # datasets are heavily nested, use nested_lookup to find 
            # the thread_items key for thread data
            thread_items = nested_lookup("thread_items", data)
            if not thread_items:
                continue
            # use our jmespath parser to reduce the dataset to the most important fields
            threads = [parse_thread(t) for thread in thread_items for t in thread]
            return {
                # the first parsed thread is the main post:
                "thread": threads[0],
                # other threads are replies:
                "replies": threads[1:],
            }
        raise ValueError("could not find thread data in page")


def convert_epoch_to_date(epoch):
    """Convert epoch date to d-m-Y string format"""
    # Convert the epoch to seconds
    epoch_in_seconds = epoch / 1000.0
    # Convert the seconds to a datetime object
    date = datetime.datetime.fromtimestamp(epoch_in_seconds)
    return date.strftime("%d-%m-%Y")


def create_dataset(threads_list):
    """ create an array with the threads and reply info """
    result = []
    for thread in threads_list:
        url = thread[0]
        sentimiento = thread[1]
        genero = thread[2]
        grupo_etario = thread[3]

        scrapped_thread = scrape_thread(url)
        result.append([
            scrapped_thread["thread"]["text"],
            convert_epoch_to_date(scrapped_thread["thread"]["published_on"]),
            sentimiento,
            # obtener_sentimiento(scrapped_thread["thread"]["text"]) if obtener_sentimiento(scrapped_thread["thread"]["text"]) else "N/A",
            genero,
            grupo_etario,
            scrapped_thread["thread"]["username"],
            "N/A",
            url,
        ])
        #  Get all the thread replies and add them to the dataset
        for reply in scrapped_thread["replies"]:
            result.append([
                reply["text"],
                convert_epoch_to_date(reply["published_on"]),
                sentimiento,
                # obtener_sentimiento(reply["text"]) if obtener_sentimiento(reply["text"]) else "N/A" ,
                "N/A",
                grupo_etario,
                reply["username"],
                "N/A",
                url,
            ])
        # time.sleep(45)

    return result


def csv_file_sorting(dataset):
    """ Get the dataset and add it to a csv with pandas"""
    file_input = pd.DataFrame(dataset)
    file_input.columns = [
        "Texto",
        "Fecha",
        "Sentimiento",
        # "Sentimiento AI",
        "Genero",
        "Grupo etario",
        "Nombre usuario",
        "Escolaridad",
        "Fuente",
    ]

    # return file_input.to_csv('dataset.csv', index=False, encoding="utf-8")
    return file_input.to_excel('output_dataset.xlsx')


def main():
    dataset = create_dataset(threads_list)
    csv_file_sorting(dataset)


if __name__ == "__main__":
    main()
    pass