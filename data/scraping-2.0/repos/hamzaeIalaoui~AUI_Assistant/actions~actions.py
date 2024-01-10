from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

import openai
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from datetime import datetime, timedelta


model = "gpt-3.5-turbo"
sheet_id = "1hA3UuDVhk2NDafuMm65lodC3VLVTjHcdyrveA1tX0Ec"
openAI_API_key = "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"


class ActionVan(Action):
    def name(self):
        return "action_van"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]):

        departure = tracker.get_slot("departure")
        
        print(tracker.latest_message.get("text"))

        if not departure:
            dispatcher.utter_message(text="Please Select a Depature below ⬇️", buttons=[
                {"payload": "I want to take the van from aui", "title": "AUI"},
                {"payload": "I want to take the van from dt", "title": "Downtown"},
                {"payload": "I want to take the van from fi", "title": "Farah Inn"},
            ])
            return []

        if departure.lower() in ("aui", "akhawayn", "al akhawayn university", "al akhawayn"):
            departure = "Al Akhawayn University"

        elif departure.lower() in ("dt", "downtown"):
            departure = "Downtown"

        elif departure.lower() in ("farah inn", "farah", "fi"):
            departure = "Farah Inn"

        else:
            dispatcher.utter_message(
                text=f"Departure Not recognized, Select one below ⬇️", buttons=[
                    {"payload": "I want to take the van from akhawayn", "title": "AUI"},
                    {"payload": "I want to take the van from dt", "title": "Downtown"},
                    {"payload": "I want to take the van from fi",
                        "title": "Farah Inn"},
                ])

        dispatcher.utter_message(text=f"{remaining_time(departure)}", buttons=[
                                 {"payload": departure, "title": "Refresh 🔄"}])

        return [SlotSet(key="departure", value=None)]


class ActionEvent(Action):
    def name(self):
        return "action_event"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ):

        latest_message = tracker.latest_message.get("text")

        print(tracker.latest_message.get("text"))


        worksheet_name = "events"
        rows = read_google_spreadsheet(sheet_id, worksheet_name)

        system = (
            "You are an AI Assistant for Al Akhawayn University students. You help students find events on campus:"
            + str(rows)
        )

        dispatcher.utter_message(text=chat_response(system, latest_message))

        return []


class ActionNavigation(Action):
    def name(self):
        return "action_navigation"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ):

        latest_message = tracker.latest_message.get("text")

        print(tracker.latest_message.get("text"))


        worksheet_name = "navigation"
        rows = read_google_spreadsheet(sheet_id, worksheet_name)
        rows2 = read_google_spreadsheet(sheet_id, "office_hours")

        system = (
            "You are an AI Assistant for Al Akhawayn University students. You help students navigate campus by guiding them until they reach their target destination and also finding professors offices:"
            + str(rows) + str(rows2)
        )

        dispatcher.utter_message(text=chat_response(
            system, latest_message), image="https://i.imgur.com/OrgkUAn.png")
        return []


class ActionOfficeHours(Action):
    def name(self):
        return "action_office_hours"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ):

        latest_message = tracker.latest_message.get("text")

        print(tracker.latest_message.get("text"))


        worksheet_name = "office_hours"
        rows = read_google_spreadsheet(sheet_id, worksheet_name)

        system = (
            "You are an AI Assistant for Al Akhawayn University students. You help students find office hours for their professors:"
            + str(rows)
        )

        dispatcher.utter_message(text=chat_response(system, latest_message))
        return []


class ActionGeneralInfo(Action):
    def name(self):
        return "action_general_info"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ):

        latest_message = tracker.latest_message.get("text")

        print(tracker.latest_message.get("text"))


        worksheet_name = "general_info"
        rows = read_google_spreadsheet(sheet_id, worksheet_name)

        system = (
            "You are an AI Assistant for Al Akhawayn University students. You help students by providing them with general information about the university, its history, its centers"
            + str(rows)
        )

        dispatcher.utter_message(text=chat_response(system, latest_message))
        return []


class ActionOpeningHours(Action):
    def name(self):
        return "action_opening_hours"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ):

        latest_message = tracker.latest_message.get("text")

        print(tracker.latest_message.get("text"))


        worksheet_name = "opening_hours"
        rows = read_google_spreadsheet(sheet_id, worksheet_name)
        rows2 = read_google_spreadsheet(sheet_id, "office_hours")

        system = (
            "You are an AI Assistant for Al Akhawayn University students. You help them find opening hours and office hours of professors of services at the university:"
            + str(rows) + str(rows2)
        )

        dispatcher.utter_message(text=chat_response(system, latest_message))
        return []


class ActionAcademicCalendar(Action):
    def name(self):
        return "action_academic_calendar"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ):

        latest_message = tracker.latest_message.get("text")

        print(tracker.latest_message.get("text"))


        worksheet_name = "academic_calendar"
        rows = read_google_spreadsheet(sheet_id, worksheet_name)

        system = (
            "You are an AI Assistant for Al Akhawayn University students. You help students by providing them with information from the academic calendar:"
            + str(rows)
        )

        dispatcher.utter_message(text=chat_response(system, latest_message))
        return []


def read_google_spreadsheet(sheet_id, worksheet_name):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "/Users/ahmedjaafari/AuiAssistant/actions/credentials.json", scope
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).worksheet(worksheet_name)
    data = sheet.get_all_values()
    headers = data.pop(0)
    rows = []
    for row in data:
        row_dict = {}
        for i in range(len(headers)):
            row_dict[headers[i]] = row[i]
        rows.append(row_dict)
    return rows


def chat_response(system, latest_message):
    openai.api_key = openAI_API_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": latest_message},
        ],
    )
    return response["choices"][0]["message"]["content"].strip()


def remaining_time(dep):
    weekday = datetime.now().weekday()
    
    dep_time= {
            "aui_to_dt": ['07:00', '07:10', '07:20', '07:30', '07:40', '07:50', '08:00', '08:10', '08:20', '08:30', '08:40', '08:50', '09:00', '09:10', '09:20', '09:30', '09:40', '09:50', '10:00', '10:10', '10:20', '10:30', '10:40', '10:50', '11:00', '11:10', '11:20', '11:30', '11:40', '11:50', '12:00', '12:10', '12:20', '12:30', '12:40', '12:50', '13:00', '13:10', '13:20', '13:30', '13:40', '13:50', '14:00', '14:10', '14:20', '14:30', '14:40', '14:50', '15:00', '15:10', '15:20', '15:30', '15:40', '15:50', '16:00', '16:10', '16:20', '16:30', '16:40', '16:50', '17:00', '17:10', '17:20', '17:30', '17:40', '17:50', '18:00', '18:10', '18:20', '18:30', '18:40', '18:50', '19:00', '19:10', '19:20', '19:30', '19:40', '19:50', '20:00', '20:10', '20:20', '21:00', '21:10', '21:20', '21:30', '21:40', '21:50', '22:00', '22:10', '22:20', '22:30', '22:40', '22:50', '23:00', '23:15', '23:45', '00:00', '00:15', '00:30', '00:45'],
            "aui_to_fi": ['07:00', '07:08', '07:16', '07:24', '07:32', '07:40', '07:48', '07:56', '08:04', '08:12', '08:20', '08:28', '08:36', '08:44', '08:52', '09:00', '09:08', '09:16', '09:24', '09:32', '09:40', '09:48', '10:00', '10:10', '10:20', '10:30', '10:40', '10:50', '11:00', '11:10', '11:20', '11:30', '11:40', '11:50', '12:00', '12:10', '12:20', '12:30', '12:40', '12:50', '13:00', '13:10', '13:20', '13:30', '13:40', '13:50', '14:00', '14:10', '14:20', '14:30', '14:40', '14:50', '15:00', '15:10', '15:20', '15:30', '15:40', '15:50', '16:00', '16:10', '16:20', '16:30', '16:40', '16:50', '17:00', '17:10', '17:20', '17:30', '17:40', '17:50', '18:00', '18:10', '18:20', '18:30', '18:40', '18:50', '19:00', '19:10', '19:20', '19:30', '19:40', '19:50', '20:00', '20:10', '20:20', '20:30', '20:40', '20:50', '21:00', '21:10', '21:20', '21:30', '21:40', '21:50', '22:00', '22:10', '22:20', '22:30', '22:40', '22:50', '23:00', '23:10', '23:20', '23:30', '23:40', '23:50', '00:00', '00:10', '00:15', '00:20', '00:25', '00:30', '00:35'],
            "Downtown": ['07:05', '07:15', '07:25', '07:35', '07:45', '07:55', '08:05', '08:15', '08:25', '08:35', '08:45', '08:55', '09:05', '09:15', '09:25', '09:35', '09:45', '09:55', '10:05', '10:15', '10:25', '10:35', '10:45', '10:55', '11:05', '11:15', '11:25', '11:35', '11:45', '11:55', '12:05', '12:15', '12:25', '12:35', '12:45', '12:55', '13:05', '13:15', '13:25', '13:35', '13:45', '13:55', '14:05', '14:15', '14:25', '14:35', '14:45', '14:55', '15:05', '15:15', '15:25', '15:35', '15:45', '15:55', '16:05', '16:15', '16:25', '16:35', '16:45', '16:55', '17:05', '17:15', '17:25', '17:35', '17:45', '17:55', '18:05', '18:15', '18:25', '18:35', '18:45', '18:55', '19:05', '19:15', '19:25', '19:35', '19:45', '19:55', '20:05', '20:15', '20:25', '21:05', '21:15', '21:25', '21:35', '21:45', '21:55', '22:05', '22:15', '22:25', '22:35', '22:45', '22:55', '23:05', '23:20', '23:50', '00:05', '00:20', '00:35', '00:50'],
            "Farah Inn": ['07:15', '07:23', '07:31', '07:39', '07:47', '07:55', '08:03', '08:11', '08:19', '08:27', '08:35', '08:43', '08:51', '08:59', '09:07', '09:15', '09:23', '09:31', '09:39', '09:47', '09:55', '10:03', '10:15', '10:25', '10:35', '10:45', '10:55', '11:05', '11:15', '11:25', '11:35', '11:45', '11:55', '12:05', '12:15', '12:25', '12:35', '12:45', '12:55', '13:05', '13:15', '13:25', '13:35', '13:45', '13:55', '14:05', '14:15', '14:25', '14:35', '14:45', '14:55', '15:05', '15:15', '15:25', '15:35', '15:45', '15:55', '16:05', '16:15', '16:25', '16:35', '16:45', '16:55', '17:05', '17:15', '17:25', '17:35', '17:45', '17:55', '18:05', '18:15', '18:25', '18:35', '18:45', '18:55', '19:05', '19:15', '19:25', '19:35', '19:45', '19:55', '20:05', '20:15', '20:25', '20:35', '20:45', '20:55', '21:05', '21:15', '21:25', '21:35', '21:45', '21:55', '22:05', '22:15', '22:25', '22:35', '22:45', '22:55', '23:05', '23:15', '23:25', '23:35', '23:45', '23:55', '00:03', '00:13', '00:23', '00:28', '00:35', '00:38', '00:43', '00:48']
        }
    
    dep_time_weekend= {
        "aui_to_fi": ['07:15', '07:45', '08:15', '08:45', '09:15', '09:45', '10:15', '10:45', '11:15', '11:45', '12:15', '12:45', '13:15', '13:45', '14:15', '15:00', '15:30', '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30', '00:00', '00:30'],
        "aui_to_dt": ['07:30', '08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30', '00:00', '00:30', '00:45'],
        "Downtown":  ['07:35', '08:05', '08:35', '09:05', '09:35', '10:05', '10:35', '11:05', '11:35', '12:05', '12:35', '13:05', '13:35', '14:05', '14:35', '15:05', '15:35', '16:05', '16:35', '17:05', '17:35', '18:05', '18:35', '19:05', '19:35', '20:05', '20:35', '21:05', '21:35', '22:05', '22:35', '23:05', '23:35', '00:05', '00:35', '00:50'],
        "Farah Inn": ['07:30', '08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:15', '15:45', '16:15', '16:45', '17:15', '17:45', '18:15', '18:45', '19:15', '19:45', '20:15', '20:45', '21:15', '21:47', '22:15', '22:45', '23:15', '23:45', '00:15', '00:45'],
    }

    if dep in ("Downtown", "Farah Inn") and weekday not in (5, 6):
        closest_departure, time_remaining, next_three = get_time(dep_time[dep])
        return f'The nearest departure from {dep} to AUI is in {time_remaining.seconds//60} mins @ {closest_departure.strftime("%H:%M %d-%m-%Y")}\nNext Departures: ' + formatFutureDepartures(next_three)
    elif dep == "Al Akhawayn University" and weekday not in (5, 6):
        closest_departure_dt, time_remaining_dt, next_three_dt= get_time(dep_time['aui_to_dt'])
        closest_departure_fi, time_remaining_fi, next_three_fi= get_time(dep_time['aui_to_fi'])
        return f'The nearest departure from AUI to Downtown is in {time_remaining_dt.seconds//60} mins @ {closest_departure_dt.strftime("%H:%M %d-%m-%Y")}\nNext Departures: ' + formatFutureDepartures(next_three_dt) + f'\n\nThe nearest departure from AUI to Farah Inn is in {time_remaining_fi.seconds//60} mins @ {closest_departure_fi.strftime("%H:%M %d-%m-%Y")}\nNext Departures: ' + formatFutureDepartures(next_three_fi)
    
    if dep in ("Downtown", "Farah Inn") and weekday in (5, 6) :
        closest_departure, time_remaining, next_three = get_time(dep_time_weekend[dep])
        return f'The nearest departure from {dep} to AUI is in {time_remaining.seconds//60} mins @ {closest_departure.strftime("%H:%M %d-%m-%Y")}\nNext Departures: ' + formatFutureDepartures(next_three)
    elif dep == "Al Akhawayn University" and weekday in (5, 6):
        closest_departure_dt, time_remaining_dt, next_three_dt= get_time(dep_time_weekend['aui_to_dt'])
        closest_departure_fi, time_remaining_fi, next_three_fi= get_time(dep_time_weekend['aui_to_fi'])
        return f'The nearest departure from AUI to Downtown is in {time_remaining_dt.seconds//60} mins @ {closest_departure_dt.strftime("%H:%M %d-%m-%Y")}\nNext Departures: ' + formatFutureDepartures(next_three_dt) + f'\n\nThe nearest departure from AUI to Farah Inn is in {time_remaining_fi.seconds//60} mins @ {closest_departure_fi.strftime("%H:%M %d-%m-%Y")}\nNext Departures: ' + formatFutureDepartures(next_three_fi)



def get_time(dep_time):
    time_format= '%H:%M'

    # Get current time and add 1 hour
    now= datetime.now()

    # convert departure times to datetime objects
    departure_datetime= [datetime.strptime(time, time_format).replace(year=now.year, month=now.month, day=now.day) for time in dep_time]

    # filter future departure times
    future_departures= list(filter(lambda dep: dep > now, departure_datetime))

    # check if there are no future departure times
    if not future_departures:
        # add a day to the departure times
        departure_datetime=list(
            map(lambda dep: dep + timedelta(days=1), departure_datetime))
        future_departures=departure_datetime

    closest_departure=min(future_departures, key=lambda x: abs(x - now))

    # calculate time remaining until closest departure
    time_remaining=closest_departure - now

    next_three = []
    
    for next_departure in future_departures[1:4]:
        next_three.append(next_departure)
    
    return closest_departure, time_remaining, next_three


def formatFutureDepartures(next_three):
    st = ""
    
    for i, a in enumerate(next_three):
        if i != 0:
            st += ", "
        st += f"{a.strftime('%H:%M')}"
    
    return st