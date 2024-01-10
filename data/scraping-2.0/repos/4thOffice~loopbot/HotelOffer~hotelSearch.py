import sys
import requests
import os
if os.path.dirname(os.path.realpath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("../")
import keys
from Auxiliary.verbose_checkpoint import verbose
from amadeus import Client, ResponseError
import googleAPI
import openai

amadeus = Client(
    client_id=keys.amadeus_client_id,
    client_secret=keys.amadeus_client_secret,
    hostname='production'
)

def translate(text):
    openai.api_key = keys.openAI_APIKEY

    user_msg = "Translate the following text to slovenian:\n\n" + text

    response = openai.chat.completions.create(
                                            model="gpt-3.5-turbo",
                                            messages=[{"role": "user", "content": user_msg}]
                                            )
    
    answer = response.choices[0].message.content
    return answer

def convert_currency(baseCurrency, currency, api_key=keys.fixer_APIKEY):
    base_url = 'http://data.fixer.io/api/latest'
    
    print(api_key)
    params = {
        'access_key': api_key,
        'base': baseCurrency,
        'symbols': currency
    }

    response = requests.get(base_url, params=params)

    # Checking if the request was successful (status code 200)
    if response.status_code == 200:
        # Getting the converted amount from the response JSON
        data = response.json()
        if data["success"] == False:
            print(data)
            print("Failed to fetch conversion data")
            return None
        print(f"Converted {baseCurrency} to {currency}")
        return data["rates"][currency]
    else:
        print("Failed to fetch conversion data. Status code:", response.status_code)
        return None


def get_access_token(api_key=keys.amadeus_client_id, api_secret=keys.amadeus_client_secret):
    auth_url = 'https://api.amadeus.com/v1/security/oauth2/token'
    response = requests.post(auth_url, data={
        'grant_type': 'client_credentials',
        'client_id': api_key,
        'client_secret': api_secret
    })
    print("Access key:\n", response.json())
    return response.json().get('access_token')

def getHotelList(access_token, cityCode, radius, stars):
    url = 'https://api.amadeus.com/v1/reference-data/locations/hotels/by-city'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/vnd.amadeus+json'
    }
    params = {
        "cityCode": cityCode,
        "radius": radius,
        "radiusUnit": "KM",
        "ratings": stars,
        "hotelSource": "ALL"
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        print('Hotel list information retrieved successfully!')

        hotelIDs = []
        hotelList = response.json()["data"]
        for hotel in hotelList:
            hotelIDs.append(hotel["hotelId"])

        return hotelIDs  # Return the JSON response
    else:
        print(f'Failed to retrieve data: {response.status_code} - {response.text}')
        return []
    
def getOfferPrice(access_token, offerID):
    url = 'https://api.amadeus.com/v3/shopping/hotel-offers/' + offerID
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/vnd.amadeus+json'
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print('Hotel offer price information retrieved successfully!')
        return response.json()
    else:
        print(f'Failed to retrieve data: {response.status_code} - {response.text}')
        return None
    
def getHotelOffers(access_token, hotelIDs, checkInDate, checkOutDate, adults, currency):
    #https://test.api.amadeus.com/v3/shopping/hotel-offers?hotelIds=YXPARVVH,UIPARWTH,FGPARIFE&adults=1&checkInDate=2024-11-22&checkOutDate=2024-11-28&roomQuantity=1&paymentPolicy=NONE&includeClosed=false&bestRateOnly=true
    url = 'https://api.amadeus.com/v3/shopping/hotel-offers'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/vnd.amadeus+json'
    }
    print("currency", currency)
    hotelIDs_ = ""
    for hotelID in hotelIDs:
        hotelIDs_ += hotelID + ","
    
    hotelIDs_ = hotelIDs_[:-1]
    print(hotelIDs_)
    params = {
        "hotelIds": hotelIDs_,
        "adults": adults,
        "checkInDate": checkInDate,
        "checkOutDate": checkOutDate,
        "roomQuantity": 1,
        "currency": "EUR",
        "paymentPolicy": "NONE",
        "includeClosed": "false",
        "bestRateOnly": "true"
    }

    print(params)

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        print('Hotel offer list information retrieved successfully!')
        return response.json()["data"]  # Return the JSON response
    else:
        print(f'Failed to retrieve data: {response.status_code} - {response.text}')
        return None

def getHotelOffer(hotelDetails, verbose_checkpoint=None):
    access_token = get_access_token()

    radius = 5
    hotelOffers = []
    for i in range(5):
        hotelsIDs = getHotelList(access_token, hotelDetails["cityCode"], radius, hotelDetails["stars"])
        #print(hotelsIDs)
        hotelsIDs = hotelsIDs[:100]
        if not hotelsIDs:
            print("No hotels found, increasing search radius..")
            radius += 10
            continue

        hotelOffers = getHotelOffers(access_token, hotelsIDs, hotelDetails["checkInDate"], hotelDetails["checkOutDate"], hotelDetails["adults"], hotelDetails["currency"])

        if hotelOffers != None and len(hotelOffers) > 0:
            print(f"Hotel offers found: {len(hotelOffers)}")
            break
            
        print("No hotel offers found, increasing search radius..")
        radius += 10
    

    if len(hotelOffers) <= 0:
        return {"price": 0, "currency": "", "checkInDate": "", "checkOutDate": "", "hotelName": ""}
    
    chosenOffer = None
    try:
        for offer in hotelOffers:
            for specificOffer in offer['offers']:
                changed = False
                if chosenOffer == None:
                    chosenOffer = specificOffer
                    changed = True
                elif chosenOffer["price"]["total"] > specificOffer["price"]["total"]:
                    chosenOffer = specificOffer
                    changed = True

        offerPrice = getOfferPrice(access_token, chosenOffer["id"])["data"]

    except Exception:
        print("Error getting hotel offer price")

    if not offerPrice:
        return {"price": 0, "currency": "", "checkInDate": "", "checkOutDate": "", "hotelName": ""}
    
    print(f"Offer price:\n {offerPrice}")

    currency = offerPrice["offers"][0]["price"]["currency"]
    total = offerPrice["offers"][0]["price"]["total"]
    checkInDate = offerPrice["offers"][0]["checkInDate"]
    checkOutDate = offerPrice["offers"][0]["checkOutDate"]
    hotelName = offerPrice["hotel"]["name"]
    descriptionANG = offerPrice["offers"][0]["room"]["description"]["text"]

    descriptionSLO = translate(descriptionANG).lower()

    if currency != hotelDetails["currency"]:
        conversion_rate = convert_currency(currency, hotelDetails["currency"])
    else:
        conversion_rate = None
    print(conversion_rate)
    if conversion_rate != None:
        converted_total = round(float(conversion_rate) * float(total), 2)
        currency = hotelDetails["currency"]
        total = converted_total

    googlePlaceID = googleAPI.get_place_id(hotelDetails["latitude"], hotelDetails["longitude"], radius*1.33, hotelName)

    photosReferenceID = googleAPI.place_details(googlePlaceID)[:3]

    return {"price": total, "currency": currency, "checkInDate": checkInDate, "checkOutDate": checkOutDate, "hotelName": hotelName, "googlePlaceID": googlePlaceID, "photosReferenceID": photosReferenceID, "descriptionANG": descriptionANG, "descriptionSLO": descriptionSLO}

#getHotelOffer({"latitude": 49.01278, "longitude": 2.55, "checkInDate": "2024-09-06", "checkOutDate": "2024-09-13"})