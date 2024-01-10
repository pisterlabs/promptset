import os
import json
import requests
import logging
import datetime
import dotenv
from openai import OpenAI
import streamlit as st
from google.cloud import bigquery
from google.cloud import storage


dotenv.load_dotenv('.env')

client = OpenAI(api_key=os.getenv('OPENAI_KEY'))

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # Add a stream handler to log to the console

# Open the JSON file and load its content into a dictionary
with open('utils/functions.json', 'r') as json_file:
    functions = json.load(json_file)


class UserQuestion:

    def __init__(_self, question):
        _self.question = question
        _self.sql = None
        _self.data = None
        _self.method = None
        _self.location = None


def upload_blob_from_memory(bucket_name, contents, destination_blob_name):
    """Uploads a file to the bucket."""

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The contents to upload to the file
    # contents = "these are my contents"

    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"
    storage_client = storage.Client(project='avalanche-analytics-project')
    bucket = storage_client.bucket(bucket_name)

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]

    destination_blob_name = f'{destination_blob_name}/{formatted_datetime}.json'
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(contents)

    logging.info(f'File uploaded to gs://{bucket_name}/{destination_blob_name}')

    print(
        f"{destination_blob_name} uploaded to {bucket_name}."
    )



@st.cache_data(ttl='24h')
def response(data, question):

    '''Generates a response to the user's question based on the data provided.'''

    system_content = ('You are a helpful assistant. Summarize the context below in relation to the user question.'
                      f' <context> {data} <context> ')

    logging.info(f"Generating response - System Context: {system_content}")

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]
    )

    return completion.choices[0].message.content


@st.cache_data(persist=True, ttl=None, show_spinner=False)
def method_selector(question):

    ''' Determines which function should be called based on the user's query.'''

    messages = [{"role": "user", "content": question}]

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=messages,
        functions=functions,
    )

    logging.info(f"Method selector - System Context: {messages}, Output: {response}")

    args = json.loads(response.choices[0].message.function_call.arguments)

    return response.choices[0].message.function_call.name, args


@st.cache_data(ttl='24h')
def query_bq_data(sql_query):
    # Initialize a BigQuery client
    client = bigquery.Client(project='avalanche-analytics-project')

    try:
        # Perform a query.
        query_job = client.query(sql_query)  # API request
        rows = query_job.result()  # Waits for query to finish
        data = [dict(row.items()) for row in rows]

        # Return the result
        return data
    except Exception as e:
        # Handle exceptions, you might want to log the error or raise it again
        logging.error(f"Error: {e}")
        return None


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Lets plan an adventure!"}]
    st.session_state['sql'] = [{"question": None, "sql_query": None}]


@st.cache_data(persist=True, ttl=None, show_spinner=False)
def location_extraction(question):
    system_content = ('You will be provided with a text, and your task is to extract the county, state, elevation, latitude, and longitude from it.'
                      'Do not attempt to answer the question. Only extract the location information.'
                      'If you are unsure return None. '
                      '#### Example ###'
                      ' Text: How much snow is at Loveland Pass?'
                      ' Response: {"county": "Clear Creek", "state": "CO", "elevation": 11900, "latitude": 39.6806, "longitude": -105.8972}'
                      '#### Example ###'
                      ' Text: What is the weather forecast for 81632?'
                      ' Response: {"county": "Eagle", "state": "CO", "elevation": 7200, "latitude": 39.6445, "longitude": -106.5933}'
                      )

    completion = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=1212,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]
    )

    output = completion.choices[0].message.content

    logging.info(f"Location extraction - System Context: {system_content}, Output: {output}")

    return output


@st.cache_data(ttl='1h')
def weather_forecast(latitude, longitude):

    logging.info(f"Getting weather_forecast for Latitude: {latitude}, Longitude: {longitude}")

    url = f'https://api.weather.gov/points/{latitude},{longitude}'
    response = requests.get(url)
    data = response.json()
    forecast = requests.get(data['properties']['forecast'])
    forecast = forecast.json()

    return forecast['properties']['periods'][:6]


@st.cache_data(ttl='24h', show_spinner=False)
def snow_depth_sql(question):
    system_content = '''Given the following SQL tables, your job is to write prompts given a user’s question.

                            CREATE TABLE `avalanche-analytics-project.production.snotel` (
                            date DATE,
                            station_name STRING,
                            state_code STRING <example: 'IL'>,
                            elevation_ft INTEGER <Should always be 3000 lower than the elevation provided in the context>,
                            latitude FLOAT,
                            longitude FLOAT,
                            county_name STRING <used to determine station county or location, example: 'Eagle'>,
                            snow_water_equivalent_in FLOAT,
                            snow_water_equivalent_median_percentage FLOAT,
                            snow_depth_in FLOAT <do not use SUM()>,
                            max_temp_degF FLOAT,
                            min_temp_degF FLOAT,
                            observed_temp_degF FLOAT,
                            snow_density_percentage FLOAT,
                            new_snow FLOAT <inches, only SUM() when GROUP BY station_name>);
                            
                            <example>
                            
                            user question: How much snow is at Loveland Pass? Additional Context:{"county": "Clear Creek", "state": "CO", "elevation": 11900}
                            Response:   WITH LatestDate AS (
                                                          SELECT MAX(Date) AS max_date
                                                          FROM `avalanche-analytics-project.historical_raw.snow-depth`
                                                          WHERE county = 'Clear Creek' AND state = 'CO' AND elevation > 8900
                                                        )
                                                        
                                                        SELECT AVG(snow_depth) AS average_snow_depth_inches
                                                        FROM `avalanche-analytics-project.historical_raw.snow-depth` AS s
                                                        JOIN LatestDate AS ld
                                                        ON s.Date = ld.max_date
                                                        WHERE s.county = 'Clear Creek' AND s.state = 'CO' AND s.elevation > 8900;
                                                                    
                            <example>

                            Use Google Standard SQL.
                            Return only the SQL query.
                            Always LIMIT results when possible.'''

    completion = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=1212,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]
    )

    return completion.choices[0].message.content


@st.cache_data(ttl='24h', show_spinner=False)
def co_field_obv (zone):

    keys_to_keep = ['observed_at', 'backcountry_zone', 'url', 'avalanche_observations', 'avalanche_observations_count',
                    'weather_observations', 'weather_observations_count',  'weather_detail',
                    'snowpack_observations_count', 'snowpack_observations', 'snowpack_detail', 'area', 'route',
                    'description', 'related_report_links']

    url = 'https://avalanche.state.co.us/api-proxy/caic_data_api?_api_proxy_uri=/api/v2/observation_reports?page=1'

    response = requests.get(url)
    data = response.json()

    week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
    filtered_list = [
        {key: data_dict.get(key, None) for key in keys_to_keep}
        for data_dict in data
        if datetime.datetime.strptime(data_dict.get('observed_at', ''), '%Y-%m-%dT%H:%M:%S.%fZ') > week_ago
    ]

    filtered_list = [dict for dict in filtered_list if dict['backcountry_zone']['title'] == zone]

    print(zone, len(filtered_list))

    if len(filtered_list) == 0:
        return 'No observations found for this zone in the last week.'

    final_list = []

    for dict in filtered_list:

        oa = dict['observed_at']
        zone = dict['backcountry_zone']['title']
        url = dict['related_report_links']['external_canonical_report']
        area = dict['area']
        route = dict['route']
        avalanche_observations = None
        weather_observation = None
        snowpack_observations = None
        weather_detail = None
        snowpack_detail = None

        if dict['avalanche_observations_count'] > 0:
            avalanche_observations = dict['avalanche_observations']

        if dict['weather_observations_count'] > 0:
            weather_observation = dict['weather_observations']

        if dict['weather_detail'] is not None:
            weather_detail = dict['weather_detail']

        if dict['snowpack_detail'] is not None:
            snowpack_detail = dict['snowpack_detail']

        if dict['snowpack_observations_count'] > 0:
            snowpack_observations = dict['snowpack_observations']

        final_list.append({'observed_at': oa, 'backcountry_zone': zone, 'url': url, 'area': area, 'route': route,
                           'avalanche_observations': avalanche_observations, 'weather_observation': weather_observation,
                           'snowpack_observations': snowpack_observations, 'weather_detail': weather_detail,
                           'snowpack_detail': snowpack_detail})

        response = f'Provide a detailed, bulleted summary of the following observations. Return the value of the "url" key as well: {final_list}'

    return response



