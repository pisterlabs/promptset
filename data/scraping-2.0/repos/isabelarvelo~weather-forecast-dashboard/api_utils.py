import requests
import pandas as pd
from location import weatherloc
import pytz
from weathercode import weatherCode
from config import OPENAI_API_KEY, TOMORROW_IO_API_KEY
import pickle

SINGLE_ICONS = [1001, 2100, 2000, 4000, 4200, 4001, 4201, 5001, 5100, 5000, 5101, 7103, 7106, 7117, 7115, 7105, 7101,
                7000, 7102, 6201, 6001, 6200, 6000, 5112, 5114, 5108, 5110, 5122, 8000]

#Importing png files scraped from Github with scrape_icons.py
with open('data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
    png_files = loaded_data

def api_call(time_period="forecast", time_step="1h", units="imperial", wlo=weatherloc("08057")):
    """
    Makes an API call to the Tomorrow.io service to retrieve weather data for a specific location.
    The type of weather data, its granularity, and the measurement units can be customized via
    the function's parameters.

    :param time_period:
    :param time_step:
    :param units:
    :type wlo: weatherloc
    """
    latitude, longitude = wlo.lat, wlo.lng

    # customizing request based on time period of interest
    if time_period == "now":
        endpoint = "https://api.tomorrow.io/v4/weather/realtime?"
        url = f"{endpoint}location={latitude},{longitude}&units={units}&apikey={TOMORROW_IO_API_KEY}"
    else:
        if time_period == "historical":
            endpoint = "https://api.tomorrow.io/v4/weather/history/recent?"
        else:
            endpoint = "https://api.tomorrow.io/v4/weather/forecast?"

        url = f"{endpoint}location={latitude},{longitude}&timesteps={time_step}&units={units}&apikey={TOMORROW_IO_API_KEY}"

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers)

    return response


def convert_to_df(json_data):
    """
    Converts json output into Pandas DataFrame
    """
    timelines = json_data['timelines']
    hourly = timelines['hourly']

    rows = []
    for hour in hourly:
        row = hour['values']
        row['time'] = hour['time']
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def convert_to_local_time(utc_time, current_timezone):
    """
    Converts time from utc time to local time
    """
    local_tz = pytz.timezone(current_timezone)
    dt = utc_time.to_pydatetime()
    local_dt = dt.astimezone(local_tz)
    return local_dt.time()


def process_data(df, wlo=weatherloc("08057")):
    """
    Processes a DataFrame containing weather data by converting timestamps to local time,
     determining sunrise and sunset times, and modifying weather codes based on the time of day.

    :param df: A DataFrame containing weather data
    :param wlo: An object of type weatherloc that contains the latitude, longitude, and timezone of the location
    :return: modified DataFrame (df) with additional columns and processed weather codes
    """
    df["time"] = pd.to_datetime(df["time"])

    current_timezone = wlo.timezone
    sunrise, sunset = wlo.get_sunrise_sunset()

    local_sunrise = convert_to_local_time(sunrise, current_timezone)
    local_sunset = convert_to_local_time(sunset, current_timezone)

    df["local_time"] = df.apply(lambda row: convert_to_local_time(row["time"], current_timezone), axis=1)

    for index, row in df.iterrows():
        if row["weatherCode"] in SINGLE_ICONS:
            df.loc[index, "weatherCode"] = int(str(row["weatherCode"]) + "0")
        else:
            if (row["local_time"] < local_sunrise) or (row["local_time"] > local_sunset):
                df.loc[index, "weatherCode"] = int(str(row["weatherCode"]) + "1")
            else:
                df.loc[index, "weatherCode"] = int(str(row["weatherCode"]) + "0")
    return df


def add_desc(df):
    """
    Creates a columns with weather descriptions based on the weathercode

    :param df:
    :return: df
    """
    if 'weatherCode_desc' not in df.columns:
        for index, row in df.iterrows():
            code = row["weatherCode"]
            df["weatherCode_desc"] = weatherCode[str(code)]
    return df


def get_image_names(df):
    """
    Creates a column of image names that will link each row of data to the appropriate weather icon
    :param df:
    :return:
    """
    image_names = []
    for code in df["weatherCode"]:
        result = [key for key, value in png_files.items() if key.startswith(str(code))]
        if result:
            image_names.append(result[0])
        else:
            image_names.append("10000_clear_small.png")
    return image_names
