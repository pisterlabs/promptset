from typing import Optional, Type
import urllib.request
import requests

from bs4 import BeautifulSoup
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun


class RtvTool(BaseTool):
    name = "Novice"
    description = "Uporabi ko želiš prebrati novice iz spleta."

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        page = urllib.request.urlopen("http://rtvslo.si")
        soup = BeautifulSoup(page.read(), "html.parser")
        aktualno = soup.find(attrs={"id": "112-1"})
        main_news = aktualno.find(attrs={"class": "xl-news"})
        news = f"{main_news.find_all('a')[2].string.strip()}\n{main_news.find('p').string.strip()}\n\n"

        for item in aktualno.find_all(attrs={"class": "rotator-title-container"}):
            title = item.find_all("a")[1].string.strip()
            if "izrael" in title.lower():
                continue
            one_news = f"{title}\n{item.find('p').string.strip()}\n\n"
            news += one_news

        return f"Trenutne aktualne novice so:\n{news}\n Uporabniku jih predstavi v obliki kratkega povzetka."


class BicikeljInfoTool(BaseTool):
    name = "Bicikelj_Info"
    description = "Uporabi za informacije o trenutnem stanju koles."

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        response = requests.get("https://api.ontime.si/api/v1/bicikelj/")
        if not response.ok():
            return "There was an error. Please try again. Bad LLM."

        station_info = response.json()["results"][0]
        station = station_info["location_name"]
        num_bikes = station_info["available_bikes"]

        return f"Na postaji {station} je na voljo {num_bikes} koles."
