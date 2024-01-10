import requests
import openai
import threading
import time
import firebase_admin
import re
from firebase_admin import firestore, credentials
from bs4 import BeautifulSoup

openai.api_key = "" 

# The overall dictionary that will contain all information about a country
dataToSubmit = {}

keyPath = "" # Private key local file path
cred = credentials.Certificate(keyPath)
app = firebase_admin.initialize_app(cred)
firestore_client = firestore.client()

def getCountryData(country, url):
    # Scrape the URL and get data about the country
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    countryData = {
        'Name': country,
        'Description': "",
        'Flag': "",
        'Categories': {
            'Cuisine': {},
            'Transportation': {},
            'Education': {},
            'Sports': {},
            'Music': {},
            'Climate': {},
        },
    }

    # Get country's description paragraph
    descText = None
    try:
        descText = soup.find('table', attrs={'class':'infobox'}).findNextSibling()
    except:
        descText = None
    

    # Call OpenAI API, wait 30 seconds and try again if requests per minute limit has been reached
    if descText != None:
        prompt_complete = False
        while not prompt_complete:
            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [
                {"role": "user", "content" : "Make a short summary of this text: " + descText.text}
                ])
                result = completion.choices[0].message.content
                if(result.startswith("This text")):
                    prompt_complete = False
                else:
                    countryData['Description'] = result
                    prompt_complete = True
            except:
                time.sleep(30)
    else:
        countryData['Description'] = ""


    # Get country's flag image link
    try:
        flagLink = "https://en.wikipedia.org/" + soup.select('a[title*="Flag of"]')[0]['href']
        flagPage = requests.get(flagLink)
        flagSoup = BeautifulSoup(flagPage.content, "html.parser")

        flagDiv = flagSoup.select('div[class=fullImageLink]')[0]
        flagSvg = flagDiv.find('a')['href']
        countryData['Flag'] = "https:" + flagSvg

    except:
        countryData['Flag'] = ""


    # Check if a specific category exists, and pull the data if it does
    # Current categories:
    #   - Cuisine
    #   - Transportation/Transport
    #   - Education
    #   - Sports
    #   - Music
    #   - Climate
    #
    if soup.find(id="Cuisine") or soup.find(id="Restaurants_and_cuisine"):
        cuisine_data = {
            'images': {},
        }
        cuisine_text = ""

        # Get the right cuisine tag
        cuisine = None
        if soup.find(id="Cuisine"):
            cuisine = soup.find(id="Cuisine").parent
        elif soup.find(id="Restaurants_and_cuisine"):
            cuisine = soup.find(id="Restaurants_and_cuisine").parent
        
        for tag in cuisine.next_siblings:
            if tag.name == 'h2' or tag.name == 'h3':
                break
            elif tag.name == 'p':
                cuisine_text += tag.text + "\n"
            elif tag.name == 'figure':
                # Get and store image
                image_link = tag.find('a')
                image_page = requests.get('https://en.wikipedia.org' + image_link['href'])
                image_soup = BeautifulSoup(image_page.content, 'html.parser')
                image_div = image_soup.find('div', attrs={'class':'fullImageLink'})
                image_source_url = image_div.find('a')['href']

                # Get and store caption
                caption_tag = tag.find('figcaption')

                caption_text = caption_tag.text
                # Remove footnote links in captions (ex. [306])
                caption_footnotes = re.findall('\[([^]]+)\]', caption_text)
                for s in caption_footnotes:
                    caption_text = caption_text.replace('[' + s + ']', '')


                cuisine_data['images'][image_source_url] = caption_text

        # Call OpenAI API, wait 30 seconds and try again if requests per minute limit has been reached
        prompt_complete = False
        while not prompt_complete:
            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [
                {"role": "system", "content" : "Make a short summary of this text: " + cuisine_text}
                ])
                cuisine_data['text'] = completion.choices[0].message.content
                prompt_complete = True
            except:
                time.sleep(30)

        countryData['Categories']['Cuisine'] = cuisine_data

    if soup.find(id="Transportation") or soup.find(id="Transport"):
        transportation_data = {
            'images': {},
        }
        transportation_text = ""

        # Get the right transporation tag
        transportation = None
        if soup.find(id="Transportation"):
            transportation = soup.find(id="Transportation").parent
        elif soup.find(id="Transport"):
            transportation = soup.find(id="Transport").parent
        
        for tag in transportation.next_siblings:
            if tag.name == 'h2' or tag.name == 'h3':
                break
            elif tag.name == 'p':
                transportation_text += tag.text + "\n"
            elif tag.name == 'figure':
                # Get and store image
                image_link = tag.find('a')
                image_page = requests.get('https://en.wikipedia.org' + image_link['href'])
                image_soup = BeautifulSoup(image_page.content, 'html.parser')
                image_div = image_soup.find('div', attrs={'class':'fullImageLink'})
                image_source_url = image_div.find('a')['href']

                # Get and store caption
                caption_tag = tag.find('figcaption')

                caption_text = caption_tag.text
                # Remove footnote links in captions (ex. [306])
                caption_footnotes = re.findall('\[([^]]+)\]', caption_text)
                for s in caption_footnotes:
                    caption_text = caption_text.replace('[' + s + ']', '')

                transportation_data['images'][image_source_url] = caption_text

        # Call OpenAI API, wait 30 seconds and try again if requests per minute limit has been reached
        prompt_complete = False
        while not prompt_complete:
            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [
                {"role": "system", "content" : "Make a short summary of this text: " + transportation_text}
                ])
                transportation_data['text'] = completion.choices[0].message.content
                prompt_complete = True
            except:
                time.sleep(30)

        countryData['Categories']['Transportation'] = transportation_data

    if soup.find(id="Education"):
        education_data = {
            'images': {},
        }
        education_text = ""

        education = soup.find(id="Education").parent
        for tag in education.next_siblings:
            if tag.name == 'h2' or tag.name == 'h3':
                break
            elif tag.name == 'p':
                education_text += tag.text + "\n"
            elif tag.name == 'figure':
                # Get and store image
                image_link = tag.find('a')
                image_page = requests.get('https://en.wikipedia.org' + image_link['href'])
                image_soup = BeautifulSoup(image_page.content, 'html.parser')
                image_div = image_soup.find('div', attrs={'class':'fullImageLink'})
                image_source_url = image_div.find('a')['href']

                # Get and store caption
                caption_tag = tag.find('figcaption')

                caption_text = caption_tag.text
                # Remove footnote links in captions (ex. [306])
                caption_footnotes = re.findall('\[([^]]+)\]', caption_text)
                for s in caption_footnotes:
                    caption_text = caption_text.replace('[' + s + ']', '')

                education_data['images'][image_source_url] = caption_text

        # Call OpenAI API, wait 30 seconds and try again if requests per minute limit has been reached
        prompt_complete = False
        while not prompt_complete:
            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [
                {"role": "system", "content" : "Make a short summary of this text: " + education_text}
                ])
                education_data['text'] = completion.choices[0].message.content
                prompt_complete = True
            except:
                time.sleep(30)

        countryData['Categories']['Education'] = education_data

    if soup.find(id="Sports") or soup.find(id="Sports_and_recreation") or soup.find(id="Sport_and_recreation"):
        sports_data = {
            'images': {},
        }
        sports_text = ""

        # Get the right sports tag
        sports = None
        if soup.find(id="Sports"):
            sports = soup.find(id="Sports").parent
        elif soup.find(id="Sports_and_recreation"):
            sports = soup.find(id="Sports_and_recreation").parent
        elif soup.find(id="Sport_and_recreation"):
            sports = soup.find(id="Sport_and_recreation").parent
        
        for tag in sports.next_siblings:
            if tag.name == 'h2' or tag.name == 'h3':
                break
            elif tag.name == 'p':
                sports_text += tag.text + "\n"
            elif tag.name == 'figure':
                # Get and store image
                image_link = tag.find('a')
                image_page = requests.get('https://en.wikipedia.org' + image_link['href'])
                image_soup = BeautifulSoup(image_page.content, 'html.parser')
                image_div = image_soup.find('div', attrs={'class':'fullImageLink'})
                image_source_url = image_div.find('a')['href']

                # Get and store caption
                caption_tag = tag.find('figcaption')

                caption_text = caption_tag.text
                # Remove footnote links in captions (ex. [306])
                caption_footnotes = re.findall('\[([^]]+)\]', caption_text)
                for s in caption_footnotes:
                    caption_text = caption_text.replace('[' + s + ']', '')

                sports_data['images'][image_source_url] = caption_text

        # Call OpenAI API, wait 30 seconds and try again if requests per minute limit has been reached
        prompt_complete = False
        while not prompt_complete:
            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [
                {"role": "system", "content" : "Make a short summary of this text: " + sports_text}
                ])
                sports_data['text'] = completion.choices[0].message.content
                prompt_complete = True
            except:
                time.sleep(30)

        countryData['Categories']['Sports'] = sports_data

    if soup.find(id="Music"):
        music_data = {
            'images': {},
        }
        music_text = ""

        music = soup.find(id="Music").parent
        for tag in music.next_siblings:
            if tag.name == 'h2' or tag.name == 'h3':
                break
            elif tag.name == 'p':
                music_text += tag.text + "\n"
            elif tag.name == 'figure':
                # Get and store image
                image_link = tag.find('a')
                image_page = requests.get('https://en.wikipedia.org' + image_link['href'])
                image_soup = BeautifulSoup(image_page.content, 'html.parser')
                image_div = image_soup.find('div', attrs={'class':'fullImageLink'})
                image_source_url = image_div.find('a')['href']

                # Get and store caption
                caption_tag = tag.find('figcaption')

                caption_text = caption_tag.text
                # Remove footnote links in captions (ex. [306])
                caption_footnotes = re.findall('\[([^]]+)\]', caption_text)
                for s in caption_footnotes:
                    caption_text = caption_text.replace('[' + s + ']', '')

                music_data['images'][image_source_url] = caption_text

        # Call OpenAI API, wait 30 seconds and try again if requests per minute limit has been reached
        prompt_complete = False
        while not prompt_complete:
            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [
                {"role": "system", "content" : "Make a short summary of this text: " + music_text}
                ])
                music_data['text'] = completion.choices[0].message.content
                prompt_complete = True
            except:
                time.sleep(30)

        countryData['Categories']['Music'] = music_data

    if soup.find(id="Climate"):
        climate_data = {
            'images': {},
        }
        climate_text = ""

        climate = soup.find(id="Climate").parent
        for tag in climate.next_siblings:
            if tag.name == 'h2' or tag.name == 'h3':
                break
            elif tag.name == 'p':
                climate_text += tag.text + "\n"
            elif tag.name == 'figure':
                # Get and store image
                image_link = tag.find('a')
                image_page = requests.get('https://en.wikipedia.org' + image_link['href'])
                image_soup = BeautifulSoup(image_page.content, 'html.parser')
                image_div = image_soup.find('div', attrs={'class':'fullImageLink'})
                image_source_url = image_div.find('a')['href']

                # Get and store caption
                caption_tag = tag.find('figcaption')

                caption_text = caption_tag.text
                # Remove footnote links in captions (ex. [306])
                caption_footnotes = re.findall('\[([^]]+)\]', caption_text)
                for s in caption_footnotes:
                    caption_text = caption_text.replace('[' + s + ']', '')

                climate_data['images'][image_source_url] = caption_text

        # Call OpenAI API, wait 30 seconds and try again if requests per minute limit has been reached
        prompt_complete = False
        while not prompt_complete:
            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [
                {"role": "system", "content" : "Make a short summary of this text: " + climate_text}
                ])
                climate_data['text'] = completion.choices[0].message.content
                prompt_complete = True
            except:
                time.sleep(30)

        countryData['Categories']['Climate'] = climate_data

    dataToSubmit[country] = countryData
    print(country + " is done")

def uploadCountryData():
    # Retrieve the entered URL, send a GET request to get HTML content and pass it to BeautifulSoup to parse through the HTML
    countryListURL = "https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population"
    page = requests.get(countryListURL)
    soup = BeautifulSoup(page.content, "html.parser")

    # Use in thread loop for testing
    testCountryURL = {
        'France': 'https://en.wikipedia.org/wiki/France',
    }

    # Create dictionary with country names linked to their respective Wikipedia URL
    countryURLS = {}
    country_table = soup.find('table', attrs={'class':'wikitable'})
    country_table_body = country_table.find('tbody')
    country_table_rows = country_table_body.find_all('tr')
    for row in country_table_rows:
        if not row.find('td'):
            continue

        current_country = row.find('td')
        if not current_country.find('a'):
            continue

        current_country_link = current_country.find('a')
        countryURLS[current_country_link.text] = "https://en.wikipedia.org/" + current_country_link['href']

    # Store each country in a separate thread for data collection
    thread_list = []
    for country in countryURLS.keys():
        thread = threading.Thread(target=getCountryData, args=(country, countryURLS[country],))
        thread_list.append(thread)
        thread.start()

    # Wait for every thread to be finished before submitting to Firebase
    for thread in thread_list:
        thread.join()
    
    # Upload country data to Firebase
    for country in dataToSubmit.keys():
        doc_ref = firestore_client.collection("countries").document(country)
        doc_ref.set(dataToSubmit[country])
        print(country + " uploaded")
    

uploadCountryData()