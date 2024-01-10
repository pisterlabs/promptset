from flask import Blueprint, render_template, request, redirect, url_for
from dotenv import load_dotenv
import requests as requests
from thefuzz import process
from collections import defaultdict
import openai
import os

load_dotenv()

ratings = Blueprint("ratings", __name__)

# Each hall key has the place_id, image, and rating (?/5) of the hall as the value.
hallIDs = {"Buxton": ["ChIJl-61hrlAwFQRSsuaxoMTm7I", "buxton.jpg", "https://www.google.com/search?q=buxton+hall+oregon+state+university&rlz=1C5CHFA_enUS984US984&sxsrf=APwXEddvTu0ijXk7hDQQp4o6FIuYjA-ipw%3A1685323106711&ei=Yv1zZL2EK9GC0PEPsuWSUA&oq=buxton+hall+oregon+state&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQARgAMgcIIxCKBRAnMgsILhCABBDHARCvAToKCAAQRxDWBBCwAzoFCAAQgAQ6BggAEBYQHjoICAAQFhAeEA86CAgAEIoFEIYDSgQIQRgAUKgNWLYVYJgZaAVwAXgAgAFviAH2B5IBBDExLjGYAQCgAQHAAQHIAQg&sclient=gws-wiz-serp#lrd=0x54c040b986b5ee97:0xb29b1383c69acb4a,1,,,,"],
        "Finley": ["ChIJ_3mYZLhAwFQR7XyBRBJgcxw", "finley.jpg", "https://www.google.com/search?q=finley+hall+osu&rlz=1C5CHFA_enUS984US984&sxsrf=APwXEdfb1uJfeaSWsxXRl9nbm2slhWRwvQ%3A1685323138932&ei=gv1zZI7BOLiS0PEP--y9iAU&gs_ssp=eJzj4tZP1zcsSTMryjIvNGC0UjWoMDVJNjAxSLIwM7G0MLdMS7MyqDBMNjc2MzA0MjGxMDRPTk3x4k_LzMtJrVTISMzJUcgvLgUAkrgT3g&oq=Finley+&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQARgAMg0ILhCKBRDHARCvARBDMg0ILhCKBRDHARCvARBDMgsILhCABBDHARCvATIFCC4QgAQyDgguEK8BEMcBEIoFEJECMgsILhCABBCxAxCDATIHCC4QigUQQzILCC4QrwEQxwEQgAQyBQguEIAEMhEILhCABBCxAxCDARDHARCvATIbCC4QigUQxwEQrwEQQxCXBRDcBBDeBBDgBBgBOgoIABBHENYEELADOgQIIxAnOgcIIxCKBRAnOg4ILhCKBRDHARCvARCRAjoRCC4QgAQQsQMQgwEQxwEQ0QM6CwgAEIAEELEDEIMBOgsILhCDARCxAxCABDoLCC4QgAQQxwEQ0QM6CwguEIMBELEDEIoFOg4ILhCABBCxAxDHARDRAzoHCAAQigUQQzoICC4QgAQQsQM6BQgAEIAEOgoILhCKBRCxAxBDOg0ILhCABBAUEIcCELEDOggIABCABBCxAzocCC4QigUQxwEQrwEQkQIQlwUQ3AQQ3gQQ4AQYAToICC4QsQMQgARKBAhBGABQpQZYtxJgjhloA3ABeACAAYABiAHtBZIBAzMuNJgBAKABAcABAcgBCNoBBggBEAEYFA&sclient=gws-wiz-serp#lrd=0x54c040b8649879ff:0x1c73601244817ced,1,,,,"], 
            "Hawley": ["ChIJgeovJ7pAwFQRNO3tCp1EiVg", "hawley.jpg", "https://www.google.com/search?q=hawley+hall+oregon+state&rlz=1C5CHFA_enUS984US984&oq=hawley+hall+&aqs=chrome.0.0i20i263i512j0i512j69i57j46i10i175i199i512j0i10i512l6.4964j0j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040ba272fea81:0x5889449d0aeded34,1,,,,"],
            "Sackett": ["ChIJnUd4OLpAwFQRSqnwlInD_CI", "sackett.jpg", "https://www.google.com/search?q=sackett+hall+ratings&rlz=1C5CHFA_enUS984US984&oq=sackett+hall+ratings&aqs=chrome..69i57j33i160l2.3895j0j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040ba3878479d:0x22fcc38994f0a94a,1,,,,"],
            "Bloss": ["ChIJhYCPksdAwFQRwiByLE24TQc", "bloss.jpg", "https://www.google.com/search?q=bloss+hall+osu&rlz=1C5CHFA_enUS984US984&oq=bloss+hall&aqs=chrome.0.0i355i512j46i175i199i512j69i57j0i512j0i22i30j0i10i22i30j0i22i30l4.2771j1j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040c7928f8085:0x74db84d2c7220c2,1,,,,"],
            "Cauthorn": ["ChIJqct4rLtAwFQRuwPGTH0atWk", "cauthorn.jpg", "https://www.google.com/search?q=cauthorn+hall&rlz=1C5CHFA_enUS984US984&oq=cauthorn+hall&aqs=chrome.0.0i355i512j46i175i199i512j0i20i263i512j0i390i650l5.1407j1j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040bbac78cba9:0x69b51a7d4cc603bb,1,,,,"],
            "Weatherford": ["ChIJrdHJd7lAwFQR5PrQRWqJZps", "weatherford.jpg", "https://www.google.com/search?q=weatherford+hall&rlz=1C5CHFA_enUS984US984&oq=weatherford+hall&aqs=chrome.0.0i355i512j46i175i199i512j0i512l2j46i175i199i512j0i22i30l5.2040j1j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040b977c9d1ad:0x9b66896a45d0fae4,1,,,,"],
            "Callahan": ["ChIJtaNhF79AwFQRUhBeSyyMOwg", "callahan.jpg", "https://www.google.com/search?q=callahan+hall&rlz=1C5CHFA_enUS984US984&oq=callahan+hall&aqs=chrome..69i57j0i512j46i20i175i199i263i512j0i512j0i10i512j46i10i175i199i512j0i10i512l4.2632j0j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040bf1761a3b5:0x83b8c2c4b5e1052,1,,,,"],
            "Poling": ["ChIJ_bgYkblAwFQRyiczeStGLuI", "poling.jpg", "https://www.google.com/search?q=polling+hall&rlz=1C5CHFA_enUS984US984&oq=polling+hall&aqs=chrome..69i57j46i10i175i199i512j0i10i512j46i10i175i199i512j0i10i512j0i22i30l4.1993j0j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040b99118b8fd:0xe22e462b793327ca,1,,,,"],
            "McNary": ["ChIJ0Rh2p79AwFQRTuzrVy45L-U", "mcnary.jpg", "https://www.google.com/search?q=mcnary+hall&rlz=1C5CHFA_enUS984US984&oq=mcnary+hall&aqs=chrome.0.0i355i512j46i175i199i512j0i20i263i512j0i512j0i22i30l5.1597j0j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040bfa77618d1:0xe52f392e57ebec4e,1,,,,"],
            "Tebeau": ["ChIJjX4nXL9AwFQR-BUdk4lWhVI", "tebeau.jpg", "https://www.google.com/search?q=tebeau+hall+osu&rlz=1C5CHFA_enUS984US984&oq=tebeau+hall&aqs=chrome.0.0i20i263i355i512j46i20i175i199i263i512j69i57j0i512l2j0i22i30j0i390i650l3.2481j1j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040bf5c277e8d:0x52855689931d15f8,1,,,,"],
            "West": ["ChIJ9R3p_rlAwFQRSkObm7RtdwM", "west.jpg", "https://www.google.com/search?q=west+hall+osu&rlz=1C5CHFA_enUS984US984&oq=West+hall+osu&aqs=chrome.0.0i512j0i22i30l5.1965j0j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040b9fee91df5:0x3776db49b9b434a,1,,,,"],
            "Dixon": ["ChIJv8Ibwb9AwFQRIwDV2rxfI4U", "dixon-lodge.jpg", "https://www.google.com/search?q=dixon+lodge&rlz=1C5CHFA_enUS984US984&oq=dixon+lodge&aqs=chrome.0.35i39i650j46i175i199i512j0i22i30j0i15i22i30j0i22i30l6.3979j0j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040bfc11bc2bf:0x85235fbcdad50023,1,,,,"],
            "ILLC": ["ChIJR_dGmcdAwFQRwdunNu8qmAE", "illc.jpg", "https://www.google.com/search?q=illc+oregon+state&rlz=1C5CHFA_enUS984US984&oq=ILLC+orego&aqs=chrome.0.0i355i512j46i175i199i512j69i57j0i390i650l2.2512j0j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040c79946f747:0x1982aef36a7dbc1,1,,,,"],
            "Halsell": ["ChIJ6_8dgLhAwFQRYclL1p0Ms8A", "halsell.jpg", "https://www.google.com/search?q=halsell+hall+osu&rlz=1C5CHFA_enUS984US984&oq=hallsell+osu&aqs=chrome.1.69i57j0i8i13i30j0i390i650l4.3519j0j9&sourceid=chrome&ie=UTF-8#lrd=0x54c040b8801dffeb:0xc0b30c9dd64bc961,1,,,,"]}
allReviews = []

# Takes each hall and feeds Google Places API the ID of the hall to get the reviews and rating of the hall
for hall, hallInfo in hallIDs.items():
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={hallInfo[0]}&fields=reviews%2Crating&key={os.getenv('GOOGLE_MAPS_API_KEY')}"
    response = requests.request("GET", url, headers={}, data={})
    reviews = response.json().get('result').get('reviews')
    for review in reviews:
        allReviews.append([review.get('rating'), review.get('text'), hall])
    hallInfo.append(response.json().get('result').get('rating'))

# Gets the summary of the reviews for each hall using OpenAI API
gptOutput = defaultdict(list)
for rating, text, hall in allReviews:
    gptOutput[hall].append(text)

for hall, reviews in gptOutput.items():
    openai.api_key = os.getenv('OPEN_AI_API_KEY')
    prompt = f"After given reviews of { hall } Hall at Oregon State University, summarize the reviews given in a maximum of 250 characters. Here are the reviews: {reviews} Again, do not exceed 250 characters in your summary."
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{'role': 'user', 'content': prompt}],
    )

    try:
        gptOutput[hall] = response.choices[0]['message']['content']
    except:
        gptOutput[hall] = "No summary available"

# Sets a route for the dorms page and passes in the necessary information collected from the API to the html file
@ratings.route('/dorms', methods=["GET", "POST"])
def dorms():
    # Only the dorms page with the hall names that are similar to the search query
    if request.method == "POST":
        search = request.form["search"]
        searchResults = process.extract(search, hallIDs.keys()) # [(hall, score), (hall, score), ...]
        searchResults = [result[0] for result in searchResults if result[1] > 60] # only saves scores above 60
        return render_template('dorms.html', allReviews=allReviews, hallIDs=hallIDs, search=search, searchResults=searchResults, hall=None)
    return render_template('dorms.html', allReviews=allReviews, hallIDs=hallIDs, hall=None)

# Route for the individual hall pages, passes in the hall name and the information collected from the Google API and OpenAI API
@ratings.route('/dorms/<hall>')
def hall(hall):
    return render_template('halls.html', hall=hall, allReviews=allReviews, hallIDs=hallIDs, gptOutput=gptOutput)
