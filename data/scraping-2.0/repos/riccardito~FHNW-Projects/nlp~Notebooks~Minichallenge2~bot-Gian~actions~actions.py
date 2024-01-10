# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import requests
from bs4 import BeautifulSoup
import pandas as pd
import copy
import re
import difflib


def create_dataframe(table):
    df = pd.DataFrame()
    string_tabelle = ""
    for i, tr in enumerate(table.findAll('tr')):
        string_tabelle += "\n"
        for j, td in enumerate(tr.findAll('td')):
            string_tabelle += td.getText().replace("\n", " ").replace("\t", "").replace("\tab", "").ljust(30)
            string_tabelle += "\t"
            df.loc[i, j] = td.getText().replace("\n", " ").replace("\t", "").replace("\tab", "")
    df.index = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    df.columns = df.iloc[0]
    df = df.iloc[1:, :]
    df.index.name = ""
    return df, string_tabelle


def read_prices(season="winter"):
    if season == "winter":
        URL = "https://www.davosklostersmountains.ch/de/mountains/winter/tarife-tickets/ski-regionalpass"

    if season == "summer":
        URL = "https://www.davosklostersmountains.ch/de/mountains/sommer/tarife-tickets"

    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    body = soup.find('body')
    # create ticket dict
    tickets = dict()
    ticket_area = body.find_all("div", attrs={'class': "pimcore_area_content-accordion-area"})
    for area in ticket_area:
        # get title of area
        title = area.find("h2").getText().replace("\\x", " ").lower()
        tickets[title] = dict()
        # get info of area
        info_text = area.find("div", attrs={'class': "wysiwyg intro text-center"}).getText().replace("\n", "")
        tickets[title]["info"] = info_text
        # read cards in area
        cards = area.find_all("div", attrs={'class': "card"})
        for card in cards:
            header = card.find("div", attrs={'class': "card-header"})
            body = card.find("div", attrs={'class': "card-body"})
            card_title = re.search(r"(\w.+\w.+)+", header.getText()).group(1).lower()
            table = body.find("tbody")
            if table:
                df, str_tbl = create_dataframe(table)
                tickets[title][card_title] = str_tbl
            else:
                tickets[title][card_title] = "Keine Informationen gefunden."
    return tickets


tickets = read_prices(season="winter")


class GetTicketTypes(Action):

    def name(self) -> Text:
        return "action_get_ticket_types"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            ticket_types = ""
            for key in tickets.keys():
                ticket_types += "\n- " + key
            dispatcher.utter_message(text="Ich habe die folgenden Ticketarten gefunden: {}".format(ticket_types))
        except:
            dispatcher.utter_message(text="Ich kann deine Frage leider aktuell nicht beantorten.")
        return []


class GetInfoOnTicketType(Action):

    def name(self) -> Text:
        return "action_get_information_on_ticket_type"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            ticket_type = tracker.get_slot('ticket_type_request')
            ticket_types = [key for key in tickets.keys()]
            ticket_type_key = difflib.get_close_matches(ticket_type, ticket_types, n=1, cutoff=0.0)[0]
            # dispatcher.utter_message(text="Original:{} Guess:{}".format(ticket_type, ticket_type_key))
            dispatcher.utter_message(
                text="Ich habe die folgenden Infos gefunden: {}".format(tickets[ticket_type_key]["info"]))
        except:
            dispatcher.utter_message(
                text="Ich kann deine Frage leider aktuell nicht beantorten (action_get_information_on_ticket_type)\nEventuell muss der Ticket-Typ genauer angegeben werden.")
        return []


class GetInfoOnTicketPrice(Action):

    def name(self) -> Text:
        return "action_get_information_on_ticket_price"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            ticket_type = tracker.get_slot('ticket_type')
            ticket_types = [key for key in tickets.keys()]
            ticket_type_key = difflib.get_close_matches(ticket_type, ticket_types, n=1, cutoff=0.0)[0]

            ticket_place = tracker.get_slot('ticket_place')
            ticket_places = [key for key in tickets[ticket_type_key].keys()]
            ticket_place_key = difflib.get_close_matches(ticket_place, ticket_places, n=1, cutoff=0.0)[0]
            dispatcher.utter_message(
                text="Ich habe die folgenden Infos für {} bei {} gefunden: {}".format(ticket_type_key, ticket_place_key,
                                                                                      tickets[ticket_type_key][
                                                                                          ticket_place_key]))
        except:
            dispatcher.utter_message(
                text="Ich kann deine Frage leider aktuell nicht beantorten (action_get_information_on_ticket_price)")
        return []


# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
import datetime
import json
import pandas as pd
from datetime import date
from datetime import datetime, timedelta
import calendar
import os
import openai


def google_translate(word):
    from google_trans_new import google_translator
    translator = google_translator()
    return translator.translate(word, lang_src='en', lang_tgt='de')


def goTrans(word):
    from translate import Translator
    translator = Translator(to_lang="German")
    translation = translator.translate(word)
    return translation


def absol_toCelsius(value):
    return round((value - 273), 1)


def weekdayfromtoday():
    my_date = date.today()
    return goTrans(calendar.day_name[my_date.weekday()])


class ActionCheckWeather(Action):

    def name(self) -> Text:
        return "action_check_weather"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # python debugger
        # import pdb
        # pdb.set_trace()

        entities_day = list(tracker.get_latest_entity_values("day"))
        entities_weekday = list(tracker.get_latest_entity_values("weekday"))
        day = None
        answer = ""
        weekdays = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
        days = ["heute", "morgen"]
        today = weekdays.index(weekdayfromtoday())
        url = "https://community-open-weather-map.p.rapidapi.com/forecast/daily"
        querystring = {"q": "davos,ch"}
        headers = {
            "X-RapidAPI-Host": "community-open-weather-map.p.rapidapi.com",
            "X-RapidAPI-Key": "486cc459f5mshc19884992659bdap1a2c40jsn26162820f766"
        }
        # try:
        #    response = requests.request("GET", url, headers=headers, params=querystring)
        # except:
        with open(
                "/home/riccardnef/PycharmProjects/nlp-projects/Notebooks/Minichallenge2/bot-Nicka/actions/src/weather.json") as json_file:
            response = json.load(json_file)

        # simple weather check for today or tomorrow
        if response is not None and len(entities_day) == 1 and len(entities_weekday) == 0:
            day = entities_day[0]
            day = days.index(day)
            answer = f"{entities_day[0].title()} ist {goTrans(response['list'][day]['weather'][0]['main'])}, die Temperatur ist {absol_toCelsius(response['list'][day]['temp']['day'])} Grad."

        # simple weather check for a weekday
        if response is not None and len(entities_day) == 0 and len(entities_weekday) == 1:
            weekday = weekdays.index(entities_weekday[0])
            futureday = weekday - today
            answer = f"Am {entities_weekday[0]} wird die Wetterlage {goTrans(response['list'][futureday]['weather'][0]['main'])} und die Temperaturen sind {absol_toCelsius(response['list'][futureday]['temp']['day'])} Grad"

        # weather check from today until a weekday, goal is to get a value:fromto, who represents the iterationamount of days to forcast
        if response is not None and len(entities_day) == 1 and len(entities_weekday) == 1:
            day = entities_day[0]
            day = days.index(day)
            weekday = weekdays.index(entities_weekday[0])
            fromto = weekday - today + 1 if weekday - today + 1 > 0 else weekday - today + 1 + 7
            for i in range(day, fromto):
                TODAY = today - i if today + i <= 6 else (today + i) - 7
                answer += f"Das Wetter am {weekdays[TODAY]} ist {goTrans(response['list'][i]['weather'][0]['main'])} , die Temperatur ist {absol_toCelsius(response['list'][i]['temp']['day'])}Grad\n"

        # weather check from weekday to weekday, extracts first weekday as first element in list
        # example => "Donnerstag bis Sonntag"
        if response is not None and len(entities_day) == 0 and len(entities_weekday) == 2:
            # Donnerstag is the first element in the list
            weekday_from = weekdays.index(entities_weekday[0])
            weekday_from -= today
            # Sonntag is the second element in the list
            weekday_to = weekdays.index(entities_weekday[1])
            weekday_to -= today
            # normal case; if the first element is before the second element
            if weekday_from > 0:
                for i in range(weekday_from, weekday_to + 1):
                    # for printing the weekdays we need to know the index of today
                    answer += f"*****normal****Das Wetter am {weekdays[i + today]} ist {goTrans(response['list'][i]['weather'][0]['main'])} , die Temperatur ist {absol_toCelsius(response['list'][i]['temp']['day'])} Grad\n"
            # special case; if the first element is after the second element; not implemented yet
            if weekday_from < 0:
                answer = "Ich kann diese Anfrage leider nicht beantworten."
            # from Montag to Montag; gives weather of the weekday back
            elif weekday_to - weekday_from == 0:
                weekday = weekdays.index(entities_weekday[0])
                futureday = weekday - today
                answer = f"Am {entities_weekday[0]} wird die Wetterlage {goTrans(response['list'][futureday]['weather'][0]['main'])} und die Temperaturen sind {absol_toCelsius(response['list'][futureday]['temp']['day'])} Grad"
        #
        # eli#f response is None:
        #    answer = f"API vermutlich überlastet: response = {response}; entities_day = {entities_day}; entities_weekday = {entities_weekday}"
        dispatcher.utter_message(answer)

        return []


class ActionCheckPisten(Action):

    def name(self) -> Text:
        return "action_check_pisten"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        if list(set(list(tracker.get_latest_entity_values("day"))))[0] == "heute":
            with open(
                    "/home/riccardnef/PycharmProjects/nlp-projects/Notebooks/Minichallenge2/bot-Nicka/actions/src/pistenbericht.json") as json_file:
                pistenbericht = json.load(json_file)
                offen = sum(pistenbericht[item] for item in pistenbericht)

            with open(
                    "/home/riccardnef/PycharmProjects/nlp-projects/Notebooks/Minichallenge2/bot-Nicka/actions/src/allpisten.json") as json_file:
                allpisten = json.load(json_file)
                total = sum(allpisten[item] for item in allpisten)

            percentperlift = [pistenbericht[item] / allpisten[item] for item in pistenbericht]
            percent = offen / total

            if percent < 0.1:
                answer = "Alle Pisten sind leider geschlossen."

            if percent == 1:
                answer = "Alle Pisten sind offen."

            if percent >= 0.5:
                minopenby50 = min(list(set(percentperlift)))
                answer = f"Es sind weniger als {int(percent * 100)} % der Pisten offen. In {allpisten[allpisten.keys()[percentperlift.index(minopenby50)]]} sind {round(minopenby50 * 100, 1)}% der Pisten offen."

            else:  # percent < 0.5
                maxopenby50 = max(list(set(percentperlift)))
                answer = f"Es sind mehr als {int(percent * 100)} % der Pisten offen. In {list(allpisten.keys())[percentperlift.index(maxopenby50)]} sind {round(maxopenby50 * 100, 1)}% der Pisten offen."
            dispatcher.utter_message(answer)
        else:  # if the user wants to check the pisten for a specific day
            dispatcher.utter_message(text="Ich kann ihnen leider nur den Pistenbericht für heute anzeigen.")
        return []


class ActionCheckEvents(Action):

    def name(self) -> Text:
        return "action_check_events"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        day = list(set(list(["heute"])))[0]  #############
        df = pd.read_csv("/home/riccardnef/PycharmProjects/nlp-projects/Notebooks/Minichallenge2/bot-Gian/actions/events.csv")
        today = date.today()
        tomorrow = datetime.today() + timedelta(days=1)
        tomorrow = tomorrow.strftime("%Y-%m-%d")

        if day == "morgen":
            now = tomorrow

        else:
            now = today

        df = df[df["date"] == now]

        if df.empty:
            answer = f"{day.title()} gibt es leider keine Events"
        else:
            answer = f"{day.title()} gibt es folgende Events:\n"
            for index, row in df.iterrows():
                answer += row["name"] + "\n"

        dispatcher.utter_message(answer)
        return []


class ActionDefaultFallback(Action):

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # python debugger
        import pdb
        #pdb.set_trace()
        text = tracker.latest_message["text"]
        #dispatcher.utter_message(str(text))

        try:

            with open("/home/riccardnef/PycharmProjects/nlp-projects/Notebooks/Minichallenge2/bot-Gian/actions/src/keys.json") as json_file:
                gptkey = json.load(json_file)["key"]
                openai.api_key = gptkey
                response = openai.Completion.create(engine="text-davinci-002", prompt=text, temperature=0.1,max_tokens=60)
                #print(response)
                result = response.choices[0].text.replace("\n", "")
                dispatcher.utter_message(text=result)
        except:
            dispatcher.utter_message(text="Ich kann ihnen leider nicht antworten.")
        return []

