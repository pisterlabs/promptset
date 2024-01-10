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

    def get_interests(self, description):
        if description == None:
            return []
        
        promt = '''
    You are a helpful assistant designed to output JSON. 
    You are given a text, that represents a event description. 
    You need to match the following interests and return it as a json array.

    ["Konzerte", "Theater", "Kino", "Sport", "Ausstellungen",
        "Familie", "Auto", "Kulinarik", "Popkultur", "Tanz",
        "Job", "Messen", "Weiterbildung", "Sprache", "Kirche",
        "Technik", "KI", "Programmieren", "MAKING", "Handwerk",
        "DIY"]

        
    return the matched interests as a json array like this:
    { 'interests': ["Konzerte", "Theater", "Kino", "Sport", "Ausstellungen"] }
    '''

        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {"role": "user", "content": promt + description},
            ],
        )

        return json.loads(response.choices[0].message.content).get("interests", [])

    def Run(self):
        try:
            OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

            url = "https://www.experimenta.science/besuchen/veranstaltungen/"
            
            
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")


            event_elements = soup.find_all("div", class_="veranstaltungen-card")

            event_element = event_elements[:2]

            for event_element in event_elements:

                card_content_element = event_element.find("div", class_="veranstaltungen-card-content")

                url = card_content_element.find("a", class_="link-intern")["href"]

                if url in self.GetKnownUrls():
                    continue

                title = html.unescape(card_content_element.find("h5").text.strip())
                description = card_content_element.find("p").text.strip()
                start_tag = card_content_element.find("div", class_="start-tag").text.strip()
                start_month = card_content_element.find("div", class_="start-monat-uhrzeit").find("strong").text.strip()
                start_month = start_month.replace("Januar", "01").replace("Februar", "02").replace("März", "03").replace("April", "04").replace("Mai", "05").replace("Juni", "06").replace("Juli", "07").replace("August", "08").replace("September", "09").replace("Oktober", "10").replace("November", "11").replace("Dezember", "12")

                interests = self.get_interests(description)

                eventObject = Event(id="", description=description,
                    interests=interests,
                    location=Location(
                        city="Heilbronn",
                        country="Germany",
                        email="info@experimenta.science",
                        name="experimenta",
                        street="Experimentaplatz",
                        telefone="",
                        zip="74072"
                    ),
                    organizer="experimenta",
                    pricing="6€",
                    start_date_time="2023-" + start_month + "-" + start_tag + "T01:01:01",
                    end_date_time="2023-" + start_month + "-" + start_tag + "T01:01:01",
                    title=title,
                    url=url,
                )

                self.SendEvent(eventObject)

        except Exception as e:
            print(e)
