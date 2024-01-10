# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

import openai

from datetime import datetime, timedelta, date

import requests
import json
import dateparser

from amadeus import Client, ResponseError
import airportsdata
from dateparser import parse
import sys

# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import SlotSet, AllSlotsReset, UserUtteranceReverted, FollowupAction
import pickle

class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Hello World!")
        return []
    

# flight
class ActionQueryFlightInfo(Action):
    def name(self) -> Text:
        return "action_query_flight_info"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            ori = tracker.get_slot("flight_departure")
            dest = tracker.get_slot("flight_destination")
            date = tracker.get_slot("flight_depart_date")
            query_result = get_flight(ori, dest, date)
            output = ""
            if(query_result == None):
                output = "No flight info available on this day" 
            else:
                for res in query_result:
                    output += res 
            print(output)
            dispatcher.utter_message(text=output)
            return [AllSlotsReset()]
        except:
            dispatcher.utter_message(text="Sorry some error occurs when querying, please provide your request again.")
            return [AllSlotsReset()]

class ValidateSimpleFlightForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_simple_flight_form"
    def validate_flight_departure(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        return {"flight_departure": slot_value}
    def validate_flight_destination(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        return {"flight_destination": slot_value}
    def validate_flight_depart_date(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        return {"flight_depart_date": slot_value}

# weather
class ActionQueryWeatherInfo(Action):
    def name(self) -> Text:
        return "action_query_weather_info"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            date = tracker.get_slot("weather_date")
            loc = tracker.get_slot("weather_location")
            output = weather_search(loc, date)
            dispatcher.utter_message(text=output)
            return [AllSlotsReset()]
        except:
            dispatcher.utter_message(text="Sorry some error occurs when querying, please provide your request again.")
            return [AllSlotsReset()]

class ValidateSimpleWeatherForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_simple_weather_form"
    def validate_weather_location(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        return {"weather_location": slot_value}
    def validate_weather_date(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        return {"weather_date": slot_value}


# hotel
class ActionQueryHotelInfo(Action):
    def name(self) -> Text:
        return "action_query_hotel_info"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            city = tracker.get_slot("addr")
            checkin_date = tracker.get_slot("arrive")
            print(f"city:{city} checkin_date:{checkin_date}")
            final_data = hotel_search(city, checkin_date)
            output = ""
            if(final_data == None):
                output = "No hotel info available on this day"
            else:
                for hotel in final_data['result']:
                    name = hotel['hotel_name']
                    price = hotel['min_total_price']
                    rating = hotel['review_score']
                    address = hotel['address']
                    output += f'Hotel Name: {name}||\nPrice: {price}||\nRating: {rating}||\nAddress: {address}, {city}||\n ---------------------------------||\n'
            print(output)
            dispatcher.utter_message(text=output)
            return [AllSlotsReset()]
        except:
            dispatcher.utter_message(text="Sorry some error occurs when querying, please provide your request again.")
            return [AllSlotsReset()]
    
class ValidateSimpleHotelForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_simple_hotel_form"
    def validate_addr(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        return {"weather_location": slot_value}
    def validate_arrive(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        return {"weather_date": slot_value}

# openai
class ActionCustomFallback(Action): 
    def name(self) -> Text:
        return "action_custom_fallback"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message['text']
        api_res = get_openai_response(user_message)
        print(api_res)
        dispatcher.utter_message(text=api_res)
        return [AllSlotsReset(), FollowupAction(name="action_listen")]
    







# get_hotel
def hotel_search(city, checkin_date_raw):
    city = "New York"
    url = "https://booking-com.p.rapidapi.com/v1/hotels/locations"
    querystring = {"name":str(city),"locale":"en-us"}

    headers = {
        "X-RapidAPI-Key": "3de4017fc6mshba19b7688371e02p121936jsnfbd063d970a2",
        "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
    }

    destID_response = requests.request("GET", url, headers=headers, params=querystring)
    data = json.loads(destID_response.text)
    dest_id = data[0]['dest_id']

    room_num = 1
    checkin_date = (dateparser.parse(checkin_date_raw)+timedelta(days=7)).strftime("%Y-%m-%d")
    next_date_obj = datetime.strptime(checkin_date, "%Y-%m-%d") + timedelta(days=1)
    checkout_date = datetime.strftime(next_date_obj, "%Y-%m-%d")
    num_adults = 2
    num_children = 2

    url = "https://booking-com.p.rapidapi.com/v1/hotels/search"
    querystring = {
        "room_number": str(room_num),
        "checkin_date":str(checkin_date),
        "checkout_date": str(checkout_date),
        "dest_id": str(dest_id),
        "adults_number": str(num_adults), 
        "children_number":str(num_children),
        "dest_type":"city", 
        "locale":"en-gb", 
        "order_by":"popularity", 
        "filter_by_currency":"USD",
        "units":"metric", 
        "page_number":"0",
        "include_adjacency":"true",
        "categories_filter_ids":"class::2, class::4, free_cancellation::1"
    }
    
    response = requests.request("GET", url, headers=headers, params=querystring)

    final_data = json.loads(response.text)
    if(len(final_data) == 0): return None
    return final_data 

# get_flight
def find_iata_code(city_name:str):
    dict = {"miami": "MIA", "chicago": "ORD"}
    if(city_name.lower() in dict.keys()): return dict[city_name.lower()]

    airports = airportsdata.load('IATA')
    for iata_code, airport in airports.items():
        if airport['city'].lower() == city_name.lower():
            return iata_code
    return None

def get_date(dstr):
    date = parse(dstr) + timedelta(days=7)
    return date.strftime("%Y-%m-%d")

def get_flight(ori, dest, date):
    amadeus = Client(
        client_id='4ORCKsq1qEo6H7gwEkv8K3UxHsChDE3x',
        client_secret='hYM8OiIhUgoPJ8kc',
    )

    _originLocationCode = find_iata_code(ori)
    _destinationLocationCode = find_iata_code(dest)
    _departureDate = get_date(date)
    print(_originLocationCode, _destinationLocationCode, _departureDate)
    try: 
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=_originLocationCode,
            destinationLocationCode=_destinationLocationCode,
            departureDate=_departureDate,
            adults=1
        )
        
        # if(len(response.data) == 0): return None
        response_data = response.data
        flights = []
        query_result = []
        for flight in response_data[:5]:
            departure = flight['itineraries'][0]['segments'][0]['departure']['iataCode']
            arrival = flight['itineraries'][0]['segments'][-1]['arrival']['iataCode']
            price = flight['price']['total']
            carrier_code = flight['itineraries'][0]['segments'][0]['carrierCode']
            flight_number = flight['itineraries'][0]['segments'][0]['number']
            departure_time = flight['itineraries'][0]['segments'][0]['departure']['at']
            arrival_time = flight['itineraries'][0]['segments'][-1]['arrival']['at']
            flight_info = {'departure': departure, 'arrival': arrival, 'price': price, 'departure_time': departure_time, 'arrival_time': arrival_time}
            flights.append(flight_info)
            flight_id = carrier_code + ' ' + flight_number
            res = flight_id + '||\n' \
                + 'Departure:'+ departure + '||\n' \
                + 'Arrival:'+ arrival + '||\n' \
                + 'Price:'+ price + '||\n' \
                + 'Departure_time:' + departure_time + '||\n' \
                + 'Arrival_time:' + arrival_time + '||\n' \
                + '--------------------------------------||\n'
            query_result.append(res)
        return query_result

    except ResponseError as error:
        print(error)

# get_weather
def weather_search(city, raw_day):
    # get the relative days offset
    today = date.today()
    print(f"raw_day = {raw_day}")
    query_date = dateparser.parse(raw_day).date()
    delta = query_date - today
    relative_days = delta.days
    # make the query
    url = "https://weatherapi-com.p.rapidapi.com/forecast.json"
    #example:check the weather in London, london changes to required cities
    querystring = {"q":city,"days":relative_days}
    headers = {
        "X-RapidAPI-Key": "eebb7cad51msh62d5499668882b4p178596jsndbec4efef724",
        "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    json_data = json.loads(response.content)
    #get the weather condition and print it
    return json_data['forecast']['forecastday'][0]['day']['condition']['text']

# get_openai
def get_openai_response(prompt):
    openai.organization = "org-vWuBsijMsE4Lky7VKKbslmcG"
    openai.api_key = "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"
    print(f'prompt: {prompt}')
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        n = 1,
        max_tokens=200
    )
    return completion.choices[0].message['content']