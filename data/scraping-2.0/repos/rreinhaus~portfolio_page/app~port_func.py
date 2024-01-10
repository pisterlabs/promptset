import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
from datashader.utils import lnglat_to_meters
from holoviews.element.tiles import EsriImagery
import holoviews as hv, pandas as pd, colorcet as cc
import numpy as np
import requests
import hvplot.pandas
import geopy.distance
from numpy import asarray
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from keras.models import load_model
from nltk.corpus import stopwords
import re,string,unicodedata
import nltk
nltk.download('stopwords')
from keras.preprocessing import text, sequence
import json 
from bs4 import BeautifulSoup
import os
import imghdr
import PyPDF2
import pytesseract
import cv2
import docx2txt
import openai
import pdftotext

hv.extension("bokeh", "matplotlib")



## Housval visualizations and machine learning model

def one_postcode_graph(df, post_code, date_from, date_to):
    """A graph that showcases all the properties sold within the date and area of the postcode"""
    
    # Dataset preparation
    one_postcode = df[df['post_code'] == post_code]
    one_postcode['date'] = pd.to_datetime(one_postcode.date)
    one_postcode = one_postcode[(one_postcode['date'] >= date_from) & (one_postcode['date'] <= date_to)]

    
    # Plot
    fig = px.scatter(
        one_postcode, 
        x="date",
        y="sale_price",
        trendline='lowess',
        title=f"Specific postcode property sale price changes over time",
        marginal_y='histogram'
    )
    
    return fig

def one_city_graph(df, city, date_from, date_to):
    """A graph that showcases all the properties sold within the date and area of the city"""

    # Dataset preparation
    one_city = df[df['city'] == city]
    one_city['date'] = pd.to_datetime(one_city.date)
    one_city = one_city[(one_city['date'] >= date_from) & (one_city['date'] <= date_to)]
    
    # Plot
    fig = px.scatter(
        one_city, 
        x="date",
        y="sale_price",
        trendline='lowess',
        title=f"Properties sold in the specific city and their sale price changes over time",
        marginal_y='histogram' 
    )
    
    
    return fig
    
def format_city_data(df,city):
    one_city = df[df['city'] == city]
    one_city['date'] = pd.to_datetime(one_city.date)
    one_city.reset_index(drop=True,inplace=True)
    one_city['string_sale']  = one_city.sale_price.astype('str')
    one_city['description'] = one_city['flat_number'] + ' ' + '- sale price is' + ' ' + one_city['string_sale']
    
    return one_city

def lat_long(df):
    lat = []
    lon = []

    for i in df['full_address']:
        url = 'https://nominatim.openstreetmap.org/'
        params = {'q': i,
        'format': 'json'}
        response = requests.get(url,params=params).json()
        if len(response) > 0:
            #print(response[0]['lat'])
            lat.append(response[0]['lat'])
            lon.append(response[0]['lon'])
        else:
            lat.append(0)
            lon.append(0)
    return lat,lon

def second_phaseofformating(df):
    lat,lon = lat_long(df)
    df['lat'] = lat
    df['lon'] = lon
    df[['lat','lon']] = df[['lat','lon']].astype(float)
    df = df[(df['lat'] != 0) & (df['lon'] != 0)]
    # to eliminate latitudes that are not located in the UK
    df = df[df['lat'] >= 50]
    df.reset_index(drop=True,inplace=True)
    
    return df

def hvplot_map(df):

    df.loc[:, 'x'], df.loc[:, 'y'] = lnglat_to_meters(df['lon'], df['lat'])
    map_tiles = EsriImagery().opts(alpha=0.5, width=700, height=480, bgcolor='black')
    plot = df.hvplot(
        'x',
        'y',
        kind='scatter',
        rasterize=False,
        c='sales_bucket',
        cnorm='eq_hist',
        colorbar=True).opts(colorbar_position='bottom')
    
    return map_tiles * plot

def foliumheatmap(df):
    # Formatting
    df_1 = df.copy()
    df_1.loc[:,'latlonrange'] = df_1.loc[:,'lat'].map(str) + '-' + df_1.loc[:, 'lon'].map(str)
    df_grouped = df_1.groupby(['latlonrange', 'lat', 'lon'])
    df_heatmap = df_grouped['sales_bucket'].agg(['count']).reset_index()
    
    
    # Mapping
    df_folium = pd.DataFrame({'Lat':df_heatmap['lat'],'Long':df_heatmap['lon'],'Count':df_heatmap['count']})
    df_folium['weight'] = df_folium['Count'] / df_folium['Count'].abs().max()

    def generateBaseMap(loc, zoom=12.4, tiles='OpenStreetMap', crs='ESPG2263'):
        return folium.Map(location=loc,
                       control_scale=True, 
                       zoom_start=zoom,
                       tiles=tiles)

    base_map = generateBaseMap([df_heatmap['lat'][0],df_heatmap['lon'][0]] )

    map_values1 = df_folium[['Lat','Long','weight']]

    data = map_values1.values.tolist()

    hm = HeatMap(data,gradient={0.1: 'black', 0.3: 'lime', 0.5: 'yellow', 0.7: 'orange', 1: 'red'}, 
                    min_opacity=0.1, 
                    max_opacity=0.9, 
                    radius=25,
                    use_local_extrema=False)#.add_to(base_map)

    return base_map.add_child(hm)

def text_toml_results(full_address, model):
    # Converting the address to lat and lon coordinates
    full_address = full_address.lower()
    lat = 0
    lon = 0
    url = 'https://nominatim.openstreetmap.org/'
    params = {'q': full_address,'format': 'json'}
    response = requests.get(url,params=params).json()
    lat = response[0]['lat']
    lon = response[0]['lon']
    
    # london check
    
    if 'london' in full_address:
        londonyesno = 1
    else:
        londonyesno = 0
    
    # Distance from London
    coords_1 = (51.50722, -0.12750)
    coords_2 = (lat, lon)
    distance = geopy.distance.geodesic(coords_1, coords_2).km
    dis_km = distance
        
    # forming a dict
    
    df_dict = {'lat':[lat],'lon':[lon],'London':londonyesno, 'distance_london':dis_km}
    
    df = pd.DataFrame.from_dict(df_dict)
    
    df['dumb'] = 1
    data = asarray([df['distance_london']])
    
    # define standard scaler
    scaler = StandardScaler()
    
    # transform data
    df[['distance_london','dumb']] = scaler.fit_transform(df[['distance_london','dumb']])
    df.drop(columns=['dumb'],inplace=True)
    
    # making a prediction
    prediction = model.predict(df.head(1))[0]
    
    # Converting the log into nominal figure
    prediction = round(np.exp(prediction))
    
    return prediction
    
## Climate Change Time Series Model

def climate_df(area):
    
    #Loading the data
    path = 'data/Environment_Temperature_change_E_All_Data_NOFLAG.csv'
    df = pd.read_csv(path,encoding='latin-1')
    
    if area in df.Area.unique():
        
        # Remove the standard deviation examples
        df = df[(df['Element'] == 'Temperature change') & (df['Area'] == area) & (df['Months'] == 'Meteorological year')]
        df = pd.DataFrame(df.mean()).reset_index(drop=False)

        # Rename columns
        df.columns=['ds', 'y']

        # Clean the df and convert to integer the year 
        df = df[3:]
        df['ds'] = df['ds'].apply(lambda x: x[1:]).astype(np.int)

        return df
    
    else:
        return 'This country is not on the list'

    
def time_predict(df, start_date, end_date, area):
    # Need to clean the df if there are any nan
    df = df[~df['y'].isnull()]
    
    # future prediction df
    future_df = pd.DataFrame(np.arange(start_date, end_date), columns=['ds'])

    # train the algo
    model = Prophet()
    model.fit(df)
    
    # make a prediction
    forecast = model.predict(future_df).loc[:, ['ds', 'yhat']]
    forecast['ds'] = forecast['ds'].apply(lambda x: x.year)
    
    # Combining the historical and future date
    full_df = pd.concat([df, forecast], axis=0)
    
    full_df['Historical Data'] = full_df['y']
    full_df['Predicted Data'] = full_df['yhat']
    
    fig = px.line(
        full_df,
        x='ds',
        y=['Historical Data', 'Predicted Data'],
        color_discrete_sequence=['black', 'red'],
        labels={'ds': "Year"},
        title=f"Average {area} Temperature Change Over Time From {start_date} Til {end_date}"
    )
    
    return fig

## Deep Learning Model For Fake/True News

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)

#Removing the stopwords from text
def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    stop.update(['Reuters','(Reuters)'])
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

#Removing the noisy text
def denoise_text_and_predict(news):
    # Load the model and set tokenize params
    loaded_model = load_model("ml_models/deep_news.h5")
    max_features = 10000
    maxlen = 300
    
    # Cleaning
    text_cleaner = remove_between_square_brackets(news)
    text_ready = remove_stopwords(text_cleaner)
    
    # Tokenizing
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(text_ready)
    tokenized_train = tokenizer.texts_to_sequences(text_ready)
    x_predict = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
    
    # Predicting
    pred = (loaded_model.predict(x_predict) > 0.5).astype("int32")
    if pred.mean() == 0:
        return 'News are valid'
    else:
        return 'News are fake'
    
## Climate Change Climatiq API

def spend_food(amount, currency):
    url = "https://beta3.api.climatiq.io/estimate"

    payload = json.dumps({
      "emission_factor": {
        "activity_id": "consumer_goods-type_food_beverages_tobacco"
      },
      "parameters": {
        "money": amount,
        "money_unit": currency
      }
    })
    headers = {
      'Authorization': 'Bearer KNVVEPRBNPMQ92QPR1KC46Z6VP3J',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    total_co2 = json.loads(response.text)['co2e']
    
    return total_co2*52


def fuel(ranges):
    url = "https://beta3.api.climatiq.io/estimate"

    payload = json.dumps({
      "emission_factor": {
        "activity_id": "commercial_vehicle-vehicle_type_hgv-fuel_source_bev-engine_size_na-vehicle_age_post_2015-vehicle_weight_gt_10t_lt_12t"
      },
      "parameters": {
        "distance": ranges,
        "distance_unit": 'km'
      }
    })
    headers = {
      'Authorization': 'Bearer KNVVEPRBNPMQ92QPR1KC46Z6VP3J',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    total_co2 = json.loads(response.text)['co2e']
    
    return total_co2 *52


def flights(amount, from_dis, to_dis, passengers=1,class_flight='economy'):
    
    airports = [from_dis, to_dis]
    airport_names = []
    
    # Webscraping the aiport names
    
    for airport in airports:
        URL_air = f'https://www.iata.org/en/publications/directories/code-search/?airport.search={airport}'
        page_air = requests.get(URL_air)
        soup = BeautifulSoup(page_air.content, "html.parser")
        job_elements = soup.find_all("table", class_="datatable")
        for job_element in job_elements:
            title_element = job_element.find_all("td")
        airport_from = str(title_element[5])
        airport_from = airport_from.replace('<td>','')
        airport_from = airport_from.replace('</td>','')
        airport_names.append(airport_from)

    # Calculating the emissions
    
    url = "https://beta3.api.climatiq.io/travel/flights"

    payload = json.dumps({
        "legs": [
            {
                "from": airport_names[0],
                "to": airport_names[1],
                "passengers": passengers,
                "class": class_flight
            }
        ]
    })
    headers = {
      'Authorization': 'Bearer KNVVEPRBNPMQ92QPR1KC46Z6VP3J',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    emission = json.loads(response.text)['co2e']
    
    # Caculating annual emissions
    emission_annual = emission * (12*amount)
        
    return emission_annual

def spend_clothing(amount, currency):
    url = "https://beta3.api.climatiq.io/estimate"

    payload = json.dumps({
      "emission_factor": {
        "activity_id": "consumer_goods-type_clothing"
      },
      "parameters": {
        "money": amount,
        "money_unit": currency
      }
    })
    headers = {
      'Authorization': 'Bearer KNVVEPRBNPMQ92QPR1KC46Z6VP3J',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    total_co2 = json.loads(response.text)['co2e']
    
    return total_co2*12

def spend_restaurant(amount, currency):
    url = "https://beta3.api.climatiq.io/estimate"

    payload = json.dumps({
      "emission_factor": {
        "activity_id": "restaurants_accommodation-type_na"
      },
      "parameters": {
        "money": amount,
        "money_unit": currency
      }
    })
    headers = {
      'Authorization': 'Bearer KNVVEPRBNPMQ92QPR1KC46Z6VP3J',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    total_co2 = json.loads(response.text)['co2e']
    
    return total_co2*12

def spend_electricity(amount):
    url = "https://beta3.api.climatiq.io/estimate"

    payload = json.dumps({
      "emission_factor": {
        "activity_id": "electricity-energy_source_biogas_corn_chp"
      },
      "parameters": {
        "energy": amount,
        "energy_unit": 'kWh'
      }
    })
    headers = {
      'Authorization': 'Bearer KNVVEPRBNPMQ92QPR1KC46Z6VP3J',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    total_co2 = json.loads(response.text)['co2e']
    
    return total_co2*12

def formatting_plot(food, fuel, flights, clothing, restaurants, electricity):
    fields = {'Fields':['Food','Fuel','Electricity','Restaurant','Clothing','Flights'],
              'Estimates(kgCO2e)':[food,fuel, electricity,restaurants, clothing,flights]}
    climate = pd.DataFrame.from_dict(fields)
    
    fig_climate = px.bar(climate, x="Fields", y="Estimates(kgCO2e)",color="Estimates(kgCO2e)")   
    fig_climate.update_layout(bargap=0.1,
                           title_text ="Your carbon emission breakdown",
                           title_x=0.5,legend_title_text='Categories')
    
    return fig_climate
  
# job search API function
  
def job_search(role, contract_type, country):
  
    # Empty lists for data
    category = []
    company = []
    title = []
    tag = []
    url_two = []
    
    # Basic information
    YOUR_APP_ID = '2b4d7f29'
    YOUR_APP_KEY = 'dbe030ecf7634d0df2fd6fa8c9c67b9a'
    role = role.replace(" ", "%20")
    url = f'https://api.adzuna.com/v1/api/jobs/{country}/search/1?app_id={YOUR_APP_ID}&app_key={YOUR_APP_KEY}&&results_per_page=50&what={role}&{contract_type}=1'
    
    # API call
    payload={}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    
    # Getting the data

    full_res = json.loads(response.text)['results']

    for num in range(len(full_res)):
      title.append(full_res[num]['title'])
      tag.append(full_res[num]['category']['tag'])
      company.append(full_res[num]['company']['display_name'])
      category.append(full_res[num]['category']['label'])
      url_two.append(full_res[num]['redirect_url'])
    
    # Creating a dataframe
    data_storage = {'Title':title,'Company Name':company, 'Category':category,'URL':url}
    full_data = pd.DataFrame.from_dict(data_storage)
    
    return full_data
  
 # Dataframe cleaner 
  
def clean_dataframe(df):
    # check for missing values
    missing_values = df.isnull().sum()
    columns_to_drop = missing_values[missing_values > (0.5 * len(df))].index
    df = df.drop(columns_to_drop, axis=1)
    
    # fill missing values
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype == "object":
                df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df[column].fillna(df[column].mean(), inplace=True)
    
    # remove duplicates
    df = df.drop_duplicates()
    
    # check for inconsistencies
    for column in df.columns:
        if len(df[column].unique()) < len(df[column]) * 0.8:
            print(f"Column {column} has inconsistencies in values.")
            df[column] = df[column].apply(lambda x: x.upper() if type(x) == str else x)
    
    # format columns
    try:
      df["date_column"] = pd.to_datetime(df["date_column"], infer_datetime_format=True)
      df["numeric_column"] = df["numeric_column"].astype(float)
    except:
      pass
    
    return df
# Testing intial Wriser code

# Check file format
def check_file_format(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        return 'pdf'
    
    elif imghdr.what(file_path) is not None:
        return 'image'
    
    elif file_extension.lower() == '.docx':
        return 'word'
    
    else:
        return 'format not supported'

# Text Extraction and Open AI API calls

def image_totext(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Apply OCR using pytesseract
    text = pytesseract.image_to_string(gray)
    
    return text

# Extracting text from pdf

def pdf_totext(pdf_path):
    
    with open(pdf_path, 'rb') as f:
        pdf = pdftotext.PDF(f)
    
    text = '\n\n'.join(pdf)
    
    return text

# Extracting text from word file

def word_totext(word_path):
    
    text = docx2txt.process(file_path)
    
    return text

openai.api_key = API_KEY_OPENAI

# Chatgpt API call
def chatgpt_call(action,document_type,language, tone, length, for_what, input_text=''):
    if input_text == '':
        prompt = f"{action} a {document_type} in {language} and {tone} tone with {length} words for {for_what}"
        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "user", "content": prompt}])
        return completion.choices[0].message['content']
    else:
        prompt_input = f"{action} a {document_type} in {language} and {tone} tone with {length} words for {for_what} using the following text {input_text}"
        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "user", "content": prompt_input}])
        return completion.choices[0].message['content']
