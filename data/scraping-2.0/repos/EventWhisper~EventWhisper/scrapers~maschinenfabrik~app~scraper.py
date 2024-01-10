import requests
from bs4 import BeautifulSoup
import re
import requests
import json
import datetime, threading
import os
from openai import OpenAI
import html

from models import Event, Location
from scraper_base import BaseScraper

class Scraper(BaseScraper):

    def location_parser(self, location):

        if "maschinenfabrik" in location.lower():
            return Location(
                city="Heilbronn",
                country="Germany",
                email="info@maschinenfabrik-hn.de",
                name="Maschinenfabrik Heilbronn",
                street="Olgastrasse 45",
                telefone="07131 2769200",
                zip="74072")
        else:
            location = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', location)
            return Location(
                city="",
                country="Germany",
                email="info@maschinenfabrik-hn.de",
                name="",
                street=location,
                telefone="07131 2769200",
                zip="")


    def parse_ics(self, instring) -> json:
        in_calendar = False
        data = {}
        for line in instring.split("\n"):
            if line.startswith("BEGIN:VEVENT"):
                in_calendar = True
            elif not in_calendar:
                continue
            elif line.startswith("END:VEVENT"):
                break
            else:
                key, value = line.split(":", 1)
                data[key] = value
        return data

    def get_event_id(self, url):
        return url.split("event_id=")[1].split("&")[0]


    def get_ics(self, url):
        response = requests.get(url)
        val = self.parse_ics(response.text)
        val["event_id"] = self.get_event_id(url)
        print(val)
        return val


    def get_event_json_list(self, url):
        response = requests.get(url)

        soup = BeautifulSoup(response.content, "html.parser")

        event_elements = soup.find_all("a", {"class": "evo_ics_nCal"})

        returnlist = []

        for item in event_elements:
            returnlist = returnlist.append(self.get_ics(item["href"]))


        print("returnlist: ", returnlist)
        return returnlist

    def Run(self):
        try:

            url="https://maschinenfabrik-hn.de/programm/"

            events = self.get_event_json_list(url)
            for idx in events:
                local = Event(
                    id="",
                    interests=[],
                    description=html.unescape(idx["DESCRIPTION"]),
                    location=self.location_parser(idx["LOCATION"]),
                    organizer="Maschinenfabrik Heilbronn",
                    pricing="0€",
                    start_date_time=idx["DTSTART"],
                    end_date_time=idx["DTEND"],
                    title=idx["SUMMARY"],
                    url=idx["URL"])
                self.SendEvent(eventObject=local)
                local.print()

        except Exception as e:
            print(e)
