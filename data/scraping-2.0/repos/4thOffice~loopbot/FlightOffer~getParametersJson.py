import datetime
import json
import sys
import time
import os

import Auxiliary.verbose_checkpoint

if os.path.dirname(os.path.realpath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import openai
import keys
import flightAuxiliary

currentYear = datetime.datetime.utcnow().year

def extractSearchParameters(emailText, offerCount, verbose_checkpoint=None):
    user_msg = "I want you to extract flight details and replace values in this parameter json:\n"
    #"includeNearAirportsAsOrigin": "false" //If it is excplicitly specified that departure can be from any of near airports, then set to "true", else leaveit as "false"
    user_msg += """{
        "currencyCode": "EUR", //Keep EUR if not specified
        "passangers": 1,
        "refundableTicket": "false", //set this to "true" only if it is excplicitly specified that ticket should have insurance for the risk of cancellation
        "changeableTicket": "false", //set this to "true" only if it is excplicitly specified that ticket should be changeable
        "maximumNumberOfConnections": 0,
        "checkedBags": 0 //amount of checked bags per person, leave 0 if not specified explicitly
        "includedAirlineCodes": "" //leave empty if not specified! must be in format (comma-seperated): "6X,7X,8X"
        "flightSegments": [ //seperate flight segments that customer is asking for. Flight are usually round-trip if not specified otherwise. If customer is asking about multiple flight offers, choose ONLY one!
                {
                    "travelClass": "ECONOMY", // ONLY choose ONE from these options and no other: ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]
                    "originLocationCode": "LJU", //If location is not specified, think logically what it could be. Location codes must be EXACTLY 3-letter IATA codes! Exactly 3 letters! This parameter must NOT be empty!
                    "alternativeOriginsCodes": "", //only alternative origins for this specific flight segment. must be in format: ["LON", "MUC"]. MUST BE AN ARRAY! Leave empty if not specified!
                    "destinationLocationCode": "PAR", //Location codes must be EXACTLY 3-letter IATA codes! Exactly 3 letters! This parameter must NOT be empty!
                    "alternativeDestinationsCodes": "", //only alternative destinations for this specific flight segment. must be in format: ["LON", "MUC"]. MUST BE AN ARRAY! Leave empty if not specified!
                    "departureDate": \"""" + str(currentYear) + """-12-09", //must be in format: YYYY-MM-DD, this value MUST be ALWAYS SET
                    "exactDepartureTime": "" //leave empty if not specified! format must be: ('00:00:00' to '23:59:59) (HH:MM:SS), Connection points dont count, only final destionation points count
                    "earliestDepartureTime": "" //leave empty if not specified! format must be: ('00:00:00' to '23:59:59) (HH:MM:SS)
                    "latestDepartureTime": "" //leave empty if not specified! format must be: ('00:00:00' to '23:59:59) (HH:MM:SS)
                    "exactArrivalTime": "" //leave empty if not specified! format must be: ('00:00:00' to '23:59:59) (HH:MM:SS), Connection points dont count, only final destionation points count
                    "earliestArrivalTime": "" //leave empty if not specified! format must be: ('00:00:00' to '23:59:59) (HH:MM:SS)
                    "latestArrivalTime": "" //leave empty if not specified! format must be: ('00:00:00' to '23:59:59) (HH:MM:SS)
                    "includedConnectionPoints": "" //must be in format: ["LON", "MUC"]. MUST BE AN ARRAY! Leave empty if not specified!
                }
        ]
}\n\n"""
    user_msg += f"Change json parameter values according to the email which I will give you. If year is not specified, use {str(currentYear)}. Location codes must be 3-letter IATA codes. You can change parameter values but you cant add new parameters. Do not leave any parameters empty, except if returnDate is not specified in email text, then you MUSt leave it empty.\n\nText to extract details from:\n"
    user_msg += emailText
    user_msg += "\n\nIf there is a specific flight written, choose that one.\n\nOnce again, output should be ONLY json and NO other text!"

    max_attempts = 2  # Maximum number of attempts
    retry_interval = 10  # Retry interval in seconds
    for attempt in range(max_attempts):
        openai.api_key = keys.openAI_APIKEY
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful robot who extracts flight details from email and provides only a json of this data as output. Output must be ONLY JSON and no other text."},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.0
        )

        if response.choices:
            try:
                flight = json.loads(response.choices[0].message.content)
            except (ValueError, json.decoder.JSONDecodeError):
                Auxiliary.verbose_checkpoint.verbose(
                    f"Failed to extract JSON {attempt=}\n{response.choices[0].message.content=}\n{emailText=}", verbose_checkpoint
                )
                if attempt == 0:
                    time.sleep(retry_interval)
                continue
            #NDC
            print(flight)
            search_params = {
                "currencyCode": flight["currencyCode"],
                "originDestinations": [],
                "travelers": [],
                "sources": [
                    "GDS"
                ],
                "searchCriteria": {
                    "maxFlightOffers": offerCount,
                    "flightFilters": {
                    "crossBorderAllowed": True,
                    "connectionRestriction": {
                        "maximumNumberOfConnections": max(flight["maximumNumberOfConnections"], 0)
                    },
                    "cabinRestrictions": [
                    ]
                    }
                }
            }

            #remove unwanted flight segments
            usedOriginLocationCodes = []
            usedDestinationLocationCodes = []
            for flightSegment in flight["flightSegments"]:
                found = False
                for code in usedOriginLocationCodes:
                    if code == flightSegment["originLocationCode"]:
                        if flightSegment in flight["flightSegments"]:
                            flight["flightSegments"].remove(flightSegment)
                        found = True
                        break
                if not found:
                    usedOriginLocationCodes.append(flightSegment["originLocationCode"])
                    found = False
                for code in usedDestinationLocationCodes:
                    if code == flightSegment["destinationLocationCode"]:
                        if flightSegment in flight["flightSegments"]:
                            flight["flightSegments"].remove(flightSegment)
                        break
                if not found:
                    usedDestinationLocationCodes.append(flightSegment["destinationLocationCode"])

            if flight["maximumNumberOfConnections"] == 0:
                search_params["searchCriteria"]["flightFilters"]["connectionRestriction"]["nonStopPreferred"] = "true"
            
            search_params["searchCriteria"]["flightFilters"]["maxFlightTime"] = 200

            for index in range(0, flight["passangers"]):
                traveler =     {
                    "id": str(index+1),
                    "travelerType": "ADULT"
                }
                search_params["travelers"].append(traveler)

            if flight["includedAirlineCodes"] != "":
                search_params["searchCriteria"]["flightFilters"]["AirlineRestrictions"] = {"includedAirlineCodes": flight["includedAirlineCodes"]}

            #print("includeNearAirportsAsOrigin:", flight["includeNearAirportsAsOrigin"])

            extraTimeframes = []
            previousOriginCode = ""
            previousDestinationCode = ""
            for index, flight_ in enumerate(flight["flightSegments"]):
                if flight_["originLocationCode"] == flight["flightSegments"][0]["originLocationCode"] and index != 0:
                    break

                if flight_["destinationLocationCode"] == previousDestinationCode or flight_["originLocationCode"] == previousOriginCode:
                    continue
                else:
                    previousOriginCode = flight_["originLocationCode"]
                    previousDestinationCode = flight_["destinationLocationCode"]

                year_from_string = int(flight_["departureDate"][:4])
                date_from_string = datetime.datetime.strptime(flight_["departureDate"], "%Y-%m-%d")
                current_date = datetime.datetime.now()
                if date_from_string < current_date:
                    flight_["departureDate"] = str(year_from_string+1) + flight_["departureDate"][4:]

                segment = {
                    "id": str(index+1),
                    "originLocationCode": flight_["originLocationCode"],
                    "originRadius": 30,
                    "destinationLocationCode": flight_["destinationLocationCode"],
                    "destinationRadius": 30,
                    "departureDateTimeRange": {
                        "date": flight_["departureDate"]
                    }
                }
                    
                if flight_["alternativeDestinationsCodes"]:
                    segment["alternativeDestinationsCodes"] = flight_["alternativeDestinationsCodes"][:2]

                if flight_["alternativeOriginsCodes"]:
                    segment["alternativeOriginsCodes"] = flight_["alternativeOriginsCodes"][:2]

                if flight_["alternativeOriginsCodes"] and flight_["alternativeDestinationsCodes"]:
                    for originCode in flight_["alternativeOriginsCodes"]:
                        for destCode in flight_["alternativeDestinationsCodes"]:
                            if originCode == destCode:
                                flight_["alternativeOriginsCodes"].remove(originCode)
                                flight_["alternativeDestinationsCodes"].remove(destCode)

                if flight_["includedConnectionPoints"]:
                    if flight_["destinationLocationCode"] in flight_["includedConnectionPoints"]:
                        flight_["includedConnectionPoints"].remove(flight_["destinationLocationCode"])
                    if flight_["originLocationCode"] in flight_["includedConnectionPoints"]:
                        flight_["includedConnectionPoints"].remove(flight_["originLocationCode"])

                    segment["includedConnectionPoints"] = flight_["includedConnectionPoints"][:2]
                    
                if "exactDepartureTime" in flight_ and flight_["exactDepartureTime"]:
                    segment["departureDateTimeRange"]["time"] = flight_["exactDepartureTime"]
                    #segment["departureDateTimeRange"]["timeWindow"] = "12H"

                #elif flight_["exactArrivalTime"] != "" and flight_["exactDepartureTime"] == "":
                    #segment["arrivalDateTimeRange"] = {"time": flight_["exactArrivalTime"]}
                    #segment["departureDateTimeRange"]["timeWindow"] = "2H"

                found = False
                for indexCabinRestriction, cabinRestriction in enumerate(search_params["searchCriteria"]["flightFilters"]["cabinRestrictions"]):
                    if cabinRestriction["cabin"] == flight_["travelClass"]:
                        search_params["searchCriteria"]["flightFilters"]["cabinRestrictions"][indexCabinRestriction]["originDestinationIds"].append(str(index+1))
                        found = True
                if not found:
                    search_params["searchCriteria"]["flightFilters"]["cabinRestrictions"].append({"cabin": flight_["travelClass"], "originDestinationIds": [str(index+1)]})

                search_params["originDestinations"].append(segment)
                #search_params["searchCriteria"]["flightFilters"]["cabinRestrictions"][0]["originDestinationIds"].append(str(index+1))

                segmentDictionary = {}
                if "earliestDepartureTime" in flight_:
                    if "T" in flight_["earliestDepartureTime"]:
                        flight_["earliestDepartureTime"] = flight_["earliestDepartureTime"].split("T")[1]
                    if flight_["earliestDepartureTime"] == "24:00:00":
                        flight_["earliestDepartureTime"] = "00:00:00"
                    segmentDictionary["earliestDepartureTime"] = flight_["earliestDepartureTime"]
                if "latestDepartureTime" in flight_:
                    if "T" in flight_["latestDepartureTime"]:
                        flight_["latestDepartureTime"] = flight_["latestDepartureTime"].split("T")[1]
                    if flight_["latestDepartureTime"] == "24:00:00":
                        flight_["latestDepartureTime"] = "00:00:00"
                    segmentDictionary["latestDepartureTime"] = flight_["latestDepartureTime"]
                if "exactDepartureTime" in flight_:
                    if flight_["exactDepartureTime"] == "24:00:00":
                        flight_["exactDepartureTime"] = "00:00:00"
                    segmentDictionary["exactDepartureTime"] = flight_["exactDepartureTime"]
                if "earliestArrivalTime" in flight_:
                    if "T" in flight_["earliestArrivalTime"]:
                        flight_["earliestArrivalTime"] = flight_["earliestArrivalTime"].split("T")[1]
                    if flight_["earliestArrivalTime"] == "24:00:00":
                        flight_["earliestArrivalTime"] = "00:00:00"
                    segmentDictionary["earliestArrivalTime"] = flight_["earliestArrivalTime"]
                if "latestArrivalTime" in flight_:
                    if "T" in flight_["latestArrivalTime"]:
                        flight_["latestArrivalTime"] = flight_["latestArrivalTime"].split("T")[1]
                    if flight_["latestArrivalTime"] == "24:00:00":
                        flight_["latestArrivalTime"] = "00:00:00"
                    segmentDictionary["latestArrivalTime"] = flight_["latestArrivalTime"]
                if "exactArrivalTime" in flight_:
                    if flight_["exactArrivalTime"] == "24:00:00":
                        flight_["exactArrivalTime"] = "00:00:00"
                    segmentDictionary["exactArrivalTime"] = flight_["exactArrivalTime"]

                if not flightAuxiliary.is_valid_time_format(segmentDictionary["earliestDepartureTime"]):
                    segmentDictionary["earliestDepartureTime"] = ""
                if not flightAuxiliary.is_valid_time_format(segmentDictionary["latestDepartureTime"]):
                    segmentDictionary["latestDepartureTime"] = ""
                if not flightAuxiliary.is_valid_time_format(segmentDictionary["latestArrivalTime"]):
                    segmentDictionary["latestArrivalTime"] = ""
                if not flightAuxiliary.is_valid_time_format(segmentDictionary["earliestArrivalTime"]):
                    segmentDictionary["earliestArrivalTime"] = ""

                if segmentDictionary["earliestDepartureTime"] == segmentDictionary["exactDepartureTime"]:
                    segmentDictionary["earliestDepartureTime"] = ""
                if segmentDictionary["latestDepartureTime"] == segmentDictionary["exactDepartureTime"]:
                    segmentDictionary["latestDepartureTime"] = ""
                if segmentDictionary["latestArrivalTime"] == segmentDictionary["exactArrivalTime"]:
                    segmentDictionary["latestArrivalTime"] = ""
                if segmentDictionary["earliestArrivalTime"] == segmentDictionary["exactArrivalTime"]:
                    segmentDictionary["earliestArrivalTime"] = ""

                extraTimeframes.append(segmentDictionary)
            
            #print(flight["includeNearAirportsAsOrigin"])
            #if "includeNearAirportsAsOrigin" in flight and flight["includeNearAirportsAsOrigin"] == "true":
            #    print("nearest airports included")
            #    search_params["originDestinations"][0]["originRadius"] = 200

            checkedbags = 0
            if "checkedBags" in flight:
                checkedbags = flight["checkedBags"]
                if checkedbags == "":
                    checkedbags = 0

            refundableTicket = False
            if "refundableTicket" in flight:
                if flight["refundableTicket"] == "true":
                    refundableTicket = True
                else:
                    refundableTicket = False

            changeableTicket = False
            if "changeableTicket" in flight:
                if flight["changeableTicket"] == "true":
                    changeableTicket = True
                else:
                    changeableTicket = False

            return search_params, extraTimeframes, checkedbags, refundableTicket, changeableTicket
        else:
            if attempt < max_attempts - 1:
                print("No response received. Retrying in {} seconds...".format(retry_interval))
                time.sleep(retry_interval)  # Wait for the specified interval before retrying
            else:
                print("Exceeded maximum attempts. No response received.")
    Auxiliary.verbose_checkpoint.verbose(f"Failed to extract json from email summary {emailText=}",
                                         verbose_checkpoint)
