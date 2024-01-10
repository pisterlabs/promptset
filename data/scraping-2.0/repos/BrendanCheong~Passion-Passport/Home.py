import pyrebase
import streamlit as st
import asyncio
import requests
from datetime import datetime
import re
import openai
import streamlit.components.v1 as components
from streamlit_pills import pills
import pycountry
import pandas as pd
import folium
from streamlit_folium import st_folium, folium_static
from bs4 import BeautifulSoup


firebaseConfig = {
  "apiKey": "AIzaSyCgRqAR-Hr5yNeG31Qgd9ROgBpAOZqRtPc",
  "authDomain": "lifehack2023-f362e.firebaseapp.com",
  "projectId": "lifehack2023-f362e",
  "storageBucket": "lifehack2023-f362e.appspot.com",
  "messagingSenderId": "141951242897",
  "appId": "1:141951242897:web:afa573c83b30d45a19e530",
  "measurementId": "G-EDRDB79V2N",
  "databaseURL": "https://lifehack2023-f362e-default-rtdb.asia-southeast1.firebasedatabase.app/"
}

def getIata() :
    df = pd.read_csv('iata.csv', usecols=[0])
    return df

def isFloat(x):
    try:
        float(x)
        return True
    except:
        return False
    
# String will look something like "x: (2.342,3.124),y: (5.2443,1.4536)"
def find_long_and_lat(s):
    lst = []
    curr_left, curr_comma, curr_right = -1, -1, -1
    for i in range(len(s)):
        if (s[i] == "("):
            curr_left = i
        elif (s[i] == ","):
            if (curr_left != -1):
                curr_comma = i
        elif (s[i] == ")"):
            if (curr_comma > curr_left):
                curr_right = i
                long = s[curr_left + 1: curr_comma]
                lat = s[curr_comma + 2: curr_right]
                if ((len(long) > 3) and (long[-1].isalpha())):
                    long = long[-3]
                if ((len(lat) > 3) and (lat[-1].isalpha())):
                    lat = lat[-3]
                if ((isFloat(long)) and (isFloat(lat))):
                    lst.append((long, lat))
                curr_left, curr_comma, curr_right = -1, -1, -1
            else:
                curr_left, curr_comma = -1, -1
    if (curr_comma > curr_left): #Check final
        long = s[curr_left + 1: curr_comma]
        lat = s[curr_comma + 2: len(s)]
        if ((len(long) > 3) and (long[-1].isalpha())):
            long = long[-3]
        if ((len(lat) > 3) and (lat[-1].isalpha())):
            lat = lat[-3]
        if ((isFloat(long)) and (isFloat(lat))):
            lst.append((long, lat))
    return lst



st.set_page_config(page_title="PassionPassport - Home", page_icon = "✈️", layout = "centered", initial_sidebar_state = "auto")
st.sidebar.title("PassionPassport")
st.sidebar.image("assets/pp_logo2.jpg", use_column_width=True)

base_url = "https://test.api.amadeus.com/v1/"


openai.api_key = st.secrets['openai_key']
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
st.session_state.authenticated = False
url = "https://test.api.amadeus.com/v1/security/oauth2/token"
data = {
    "grant_type": "client_credentials",
    "client_id": st.secrets["amadeus_key"],
    "client_secret": st.secrets["amadeus_secret"]
}

response = requests.post(url, data=data)
access_token = response.json()["access_token"]

headers = {
    "Authorization": "Bearer " + access_token,
    "Content-Type": "application/json"
}


db = firebase.database()
st.session_state.db = db
storage = firebase.storage()




# Authentication
authenticate = st.sidebar.selectbox('Login/Signup', ['Login', 'Signup'])


if authenticate == 'Signup':
    email = st.sidebar.text_input('Enter your email address')
    password = st.sidebar.text_input('Enter your password', type = 'password')
    name = st.sidebar.text_input('Enter your Name')
    place = st.sidebar.text_input('Enter your Location')
    handle = st.sidebar.text_input('Please input your name', value = " ")
    submit = st.sidebar.button('Log In')


    if submit :
        user = auth.create_user_with_email_and_password(email, password)
        st.success('Your account has successfully been created!')
       
        #Signing the user and create a user variable
        user = auth.sign_in_with_email_and_password(email, password)
        db.child(user['localId']).child("Handle").set(handle)
        db.child(user['localId']).child("ID").set(user['localId'])
        db.child(user['localId']).child("Location").set(place)
        st.title("Hi, " + handle)
        st.info('Login successful')




if authenticate == 'Login' :
    email = st.sidebar.text_input('Enter your email address')
    password = st.sidebar.text_input('Enter your password', type = 'password')
    submit = st.sidebar.button('Log In')
    if submit :
        try: 
            user = auth.sign_in_with_email_and_password(email, password)
            username = db.child(user['localId']).child("Handle").get()
            st.session_state.name = username
            st.session_state.user = user
            if user is not None :
                st.session_state.authenticated = True
            st.success("Successfully signed in")
        except Exception as e : 
                st.info(e)
        else :    
            st.write("Welcome")
    openai.api_key = st.secrets['openai_key']
    st.image("https://thumbs.dreamstime.com/b/travel-banner-landmarks-airplane-around-world-tourism-background-vector-illustration-109951098.jpg")
    st.title("Passion Passport Travel Machine")
    st.write("""This is the travel machine that suggests you the best places according to your preference.
    
Not sure where to travel? Plan an itinerary with us today!""")
    people_array = ""
    option = st.selectbox('Depart from',options=getIata())[0:4]
    adults = st.number_input("Number of adults ", step=1, value=1, key="adult")
    childrens = st.number_input("Number of children ", step=1, value=0, key="child")
    childrens_string = ""
    if childrens > 0 :
        childrens_string = "&children=" + str(childrens)
    st.subheader("Preference Indicator")
    st.write("""Tell us what you like to do - your hobby, your interest, your preference
    
Sample answer: Hike the mountain, swim in the beach, be in a cold country""")
    for i in range(0, int(adults) + int(childrens)) :
        user_input = st.text_area("Interest of Person " + str(i + 1) ,placeholder = "Your suggestion", key="input" + str(i))
        st.write("Tell us what you like to do - your hobby, your interest, your preference")
        people_array += "People " + str(i + 1) + " likes to " + user_input + "."
        chat_gpt = people_array + "Suggest 1 city according to these people in the format 'Bangkok'. City only "
    date = st.date_input("Departure Date *(must be after this date)")
    
    hotelDays = st.number_input("Number of Days ", step=1, value=1, key="hotel")
    hotel_string = pd.to_datetime(date) + pd.DateOffset(days=hotelDays)
    date_string = "&returnDate=" + str(hotel_string.date().strftime("%Y-%m-%d"))
    
    current_date_valid = datetime.now().date() >= date
    pressed = st.button('Submit', disabled=current_date_valid)
        
    res_box = st.empty()
    if pressed:
        completions = openai.ChatCompletion.create(model="gpt-4", messages=[
                            {"role": "assistant",
                            "content": chat_gpt,
                            }],
                            temperature=0.5,
                            max_tokens=1000,
                            frequency_penalty=0.0,)
        
        result = completions.choices[0].message.content
        array = result.split(", ")
        st.header("Suggested Place: " + result)
        for answer in array :
            data = requests.get("https://test.api.amadeus.com/v1/reference-data/locations/cities?keyword=" + answer, headers=headers)
            
            res = (data.json()['data'][0])
            dataTwo = requests.get("https://test.api.amadeus.com/v2/shopping/flight-offers?originLocationCode=" + option[0:4] +"&destinationLocationCode=" + res['iataCode'] +"&departureDate=" + str(date) + "&adults=" + str(int(adults)) + childrens_string + date_string + "&currencyCode=SGD&max=2", headers=headers)
            dataThree = requests.get("https://test.api.amadeus.com/v2/duty-of-care/diseases/covid19-area-report?countryCode=" + res['address']['countryCode'], headers=headers)
            st.session_state.covid = dataThree.json()['data']
            resTwo = dataTwo.json()['data']
            st.session_state.flight = resTwo
            for ansTwo in resTwo :
                st.subheader("Estimated price")
                st.write("For more detailed price, visit the Price tab")
                st.text("SGD " + ansTwo['price']['total'])
            secondPrompt = "Plan me an itinerary of " + answer + "to do if" + people_array  + ". Limit to 50 words each day for " + str(hotelDays) +" days"
            completions = openai.ChatCompletion.create(model="gpt-4", messages=[
                            {"role": "assistant", 
                            "content": secondPrompt,
                            }],
                            temperature=0.2,
                            max_tokens=1000,
                            frequency_penalty=0.0,)
            secondResult = completions.choices[0].message.content
            st.subheader("Suggested itinerary")
            st.write("For more detailed itinerary, visit the Results tab")
            st.write(secondResult)
            st.session_state.itin = secondResult
            thirdResult = ''
            response2 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                            {"role": "assistant", 
                            "content": "Extract the places' longitude and latitude from this intinerary:" + secondResult + " in the strict number format without degree '(longitude, latitude)'",
                            }],
                temperature=0,
                max_tokens=64,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            thirdResult = response2.choices[0].message.content
            thirdArray = find_long_and_lat(thirdResult)
            mapData = []
            for long, lat in thirdArray:
                if ((isFloat(long)) and (isFloat(lat))):
                    try :
                        i = 0
                        mama = requests.get("https://test.api.amadeus.com/v1/shopping/activities?longitude=" + long + "&latitude=" + lat + "&radius=100", headers=headers)
                        for ans in mama.json()['data'] :
                            components.html('<h2 style="font-weight: 800; font-size: 16px; font-family: sans-serif;">' + ans['name'] + "</h2>")
                            # Create two columns with different widths
                            col1, col2 = st.columns([1, 2.5])

                            # Display the image in the first column
                            with col1:
                                if len(ans['pictures']) != 0:
                                    st.image(ans['pictures'][0], width=200)  # Adjust the width as per your requirement

                            # Display other content in the second column
                            with col2:
                                if 'shortDescription' in ans:
                                    desc = ans['shortDescription']
                                    soup1 = BeautifulSoup(desc, "html.parser")
                                    clean_text1 = soup1.get_text()
                                    st.write(desc)

                            object = {
                                    'name': ans['name'],
                                    'latitude': lat,
                                    'longitude': long,
                                    'safe': str(dataThree.json()['data']['summary']['text']),
                                    'bookingLink': ans['bookingLink'],
                                    'price': ans['price']['amount']
                            }
                            if object not in mapData :
                                mapData.append(object)
                    except :
                        continue
            
            
            
            try:
                st.header("Analysis")

                # Convert latitude and longitude columns to numeric format
                dataMap = pd.DataFrame(mapData)
                dataMap['latitude'] = pd.to_numeric(dataMap['latitude'])
                dataMap['longitude'] = pd.to_numeric(dataMap['longitude'])
                
                #st.write(dataMap)
                st.dataframe(dataMap)
                st.session_state.mapData = mapData
            except Exception as e : 
                st.info(e)


        


             


