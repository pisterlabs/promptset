import requests
import os
from openai import OpenAI
import json
import folium
import pandas as pd
from twilio.rest import Client
import webbrowser
import time


# client = OpenAI(api_key=os.environ["OPENAI_KEY"])
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def open_map():
    url = "file:///Users/mathemagie/mesprojets/sheet_sciencespo/map.html"
    webbrowser.open(url, new=2)  # open in new tab


# create a map with folium library with lat, long from resp
def create_map():
    print("create map with folium")
    resp = get_data_from_sheet()
    m = folium.Map(location=[48.8534, 2.3488], zoom_start=12)
    for i in resp:
        if i["lat"] != "" and i["lon"] != "":
            folium.Marker([i["lat"], i["lon"]], popup=i["comment"]).add_to(m)
        m.save("map.html")


def send_sms():
    print("send SMS")
    account_sid = "XXXXXXX"
    auth_token = "XXXXXX"
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        from_="XXXX", body="DONE", to="XXXXX"
    )

    print(message.sid)


def get_sentiment(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a tweet, and your task is to classify its sentiment as positive, neutral, or negative.",
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    res = response.model_dump_json()
    data = json.loads(res)
    message_content = data["choices"][0]["message"]["content"]

    return message_content  # Outputs: positive


# url = "https://api.sheety.co/75b80a29977172c897e968cc7509c3f6/apiSciencesPo/sheet1"
url = "https://api.sheety.co/917a493d299bb9a04c196c7cf0393353/testSciencesPo2/sheet1"


def get_adress_postal(lon, lat):
    url = "https://api.geoapify.com/v1/geocode/reverse?lat={}&lon={}&format=json&apiKey=XXXX".format(
        lat, lon
    )
    r = requests.get(url)
    return r.json()["results"][0]["formatted"]


def get_lat_long_from_adress(adress):
    url = "https://api.geoapify.com/v1/geocode/search?text={}&format=json&apiKey=XXXX".format(
        adress
    )
    r = requests.get(url)
    return r.json()["results"][0]["lon"], r.json()["results"][0]["lat"]


def get_trad_in_french(text):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = "auth_key=xxxx&text={}&target_lang=FR".format(
        text
    )
    response = requests.post(
        "https://api-free.deepl.com/v2/translate", headers=headers, data=data
    )
    return response.json()["translations"][0]["text"]


def edit_row(id, key, text):
    """
    Edit a row in the specified sheet with the given id.

    Args:
        id (str): The id of the sheet.
        key (str): The key of the column to be edited.
        text (str): The new text to be inserted in the column.

    Returns:
        None
    """
    data = {
        "sheet1": {
            key: text,
        }
    }
    endpoint = f"{url}/{id}"

    response = requests.put(url=endpoint, json=data)
    # print("response.status_code =", response.status_code)
    # print("response.text= ", response.text)


def get_data_from_sheet():
    r = requests.get(url)
    return r.json()["sheet1"]


def update_sheet():
    resp_data = get_data_from_sheet()
    for line in resp_data:
        id_line = line["id"]
        comment = line["comment"]
        address = line["adressePostale"]
        lon, lat = get_lat_long_from_adress(address)
        edit_row(id_line, "lon", lon)
        edit_row(id_line, "lat", lat)
        sentiment = get_sentiment(comment)
        print(f"- traitement de : {id_line} de la feuille excel")
        print(f"- lon  {lon} lat {lat}  de l'adresse : {address}")
        print(f"le sentiment du commentaire : {sentiment}")
        edit_row(id_line, "sentiment", sentiment)
        # adress = get_adress_postal(i["lon"], i["lat"])
        # print(f"l'aadresse postale de {long} {lat} : {adress}")
        # edit_row(i["id"], "adressePostale", adress)
        trad = get_trad_in_french(line["comment"])
        print(f"la traduction en français : {trad}")
        edit_row(id_line, "traduction", trad)
        print()
        print("------------")
        print()


update_sheet()

create_map()

time.sleep(3)

open_map()

#send_sms()
