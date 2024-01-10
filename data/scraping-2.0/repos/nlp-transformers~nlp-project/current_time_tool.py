from langchain.tools import Tool
import requests

def get_current_time(location) :
    IPGEOLOCATION_KEY = '' # put your IPGEOLOCATION_KEY here
    if location == "current location":
        payload = {'apiKey': IPGEOLOCATION_KEY}
    else:
        payload = {'apiKey': IPGEOLOCATION_KEY, 'location': location}

    response = requests.get('https://api.ipgeolocation.io/timezone', params=payload).json()
    if "date_time_txt" in response:
        result = "The time in "+ location + " is " + response["date_time_txt"]
    else:
        result = "Could not find time in"+ location


    return result

current_time_tool = Tool.from_function(
    func = get_current_time,
    name = "current_time",
    return_direct=True,
    description="Use this tool to obtain current time at given location. Input to tool should be the location. If location is not known, input should be 'current location'.",
)