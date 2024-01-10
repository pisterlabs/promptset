from langchain.prompts import PromptTemplate
import requests as requests
import json as json
import os



    
    
    
def getLocationDetailsFromPointCoordinates(latitude, longitude):
    """ Here is the explanation for getLocationDetailsFromPointCoordinates(latitude,longitude):
    1. We get the latitude and longitude from the user.
    2. We get the BING_MAPS_KEY from the environment variables.
    3. We build the URL for the request.
    4. We make the request.
    5. We check the status code. If it is 200, we get the result.
    6. If the status code is not 200, we return an error message. 
    """
    key = os.environ.get('BING_MAPS_KEY')
    url = "http://dev.virtualearth.net/REST/v1/Locations/" + str(latitude) + "," + str(longitude) + "?i&verboseplacenames=true&key=" + str(key)
    r = requests.get(url, auth=('user', 'pass'))
    status = r.status_code
    if status == 200:
        result = r.json()
        resultObj = result["resourceSets"][0]["resources"][0]
        return resultObj
    if status != 200:
        return "Error: " + str(status)
    

#city or neighborhood
def getAdminDistrctFromResultObject(resultObj):
    return resultObj["address"]["locality"]

## State
def getAdminDistrctFromResultObject(resultObj):
    #print("resultObj",resultObj)
    return resultObj["address"]["adminDistrict"]

## County    
def getCountryFromResultObject(resultObj):
    return resultObj["address"]["countryRegion"]

## full street address  
def getAddressFromResultObject(resultObj):
    return resultObj["address"]["formattedAddress"]


## given lat/long return state and country using bing maps API
def getStateAndCountyFromLatLong(latitude,longitude):
    resultObj = getLocationDetailsFromPointCoordinates(latitude,longitude)
    state = getAdminDistrctFromResultObject(resultObj)
    country = getCountryFromResultObject(resultObj)
    return {"state":state, "country":country}

def getAddressFromLatLong(latitude,longitude):
    resultObj = getLocationDetailsFromPointCoordinates(latitude,longitude)
    return getAddressFromResultObject(resultObj)

########## Semantic prompts

extractCityFromStreetAddress = PromptTemplate(
    input_variables=["address"],
    template="Given the following street address that could be in any country worldwide and comes from Bing Maps API, what is the likely city, town, or village {address}. Return as a single string with just that information an address, and nothing else."
)


def getPointLocationFromCityStateAndCounty(city="",state="",country=""):
    #### Get a point location from a city, state, and country in lat/long
    #### using the bing maps API
    key = os.environ.get('BING_MAPS_KEY')
    url = "http://dev.virtualearth.net/REST/v1/Locations/" + str(city) + "," + str(state) + "," + str(country) + "?i&verboseplacenames=true&key=" + str(key)
    r = requests.get(url, auth=('user', 'pass'))
    status = r.status_code
    if status == 200:
        result = r.json()
        resultObj = result["resourceSets"][0]["resources"][0]
        return resultObj
    if status != 200:
        return "Error: " + str(status)