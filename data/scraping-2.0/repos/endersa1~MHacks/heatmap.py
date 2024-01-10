import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import fitbit
import matplotlib.dates as mdates
import arrow  # Arrow is a really useful date time helper library
from openai import OpenAI
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import firebase_admin
from firebase_admin import storage
from PIL import Image
import json
import csv

# Load CSV data
df = pd.read_csv('eye_gaze_coordinates.csv')

# Extract relevant columns
scatter_data = df[['Normalized X', 'Normalized Y']]

# Create a scatter plot using Matplotlib
fig, ax = plt.subplots()
ax.scatter(scatter_data['Normalized X'], scatter_data['Normalized Y'])
ax.set_xlabel('Normalized X')
ax.set_ylabel('Normalized Y')
ax.set_title('Eye Gaze Scatter Plot')

# Display the scatter plot using Streamlit
st.pyplot(fig)


def display_heatmap(csv_path, title):
    # Load CSV data
    df = pd.read_csv(csv_path)

    # Extract relevant columns
    scatter_data = df[['Normalized X', 'Normalized Y']]

    # Calculate kernel density estimate for the data
    kde = gaussian_kde(scatter_data.T)

    # Create a grid of coordinates for the heatmap
    x_grid, y_grid = np.mgrid[scatter_data['Normalized X'].min():scatter_data['Normalized X'].max():100j,
                              scatter_data['Normalized Y'].max():scatter_data['Normalized Y'].min():100j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = np.reshape(kde(positions).T, x_grid.shape)

    # Create a heatmap without scatter plot
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(z), cmap=plt.cm.viridis, extent=[scatter_data['Normalized X'].min(),
                                                       scatter_data['Normalized X'].max(),
                                                       scatter_data['Normalized Y'].max(),
                                                       scatter_data['Normalized Y'].min()],
              alpha=0.5, aspect='auto')
    ax.set_xlabel('Normalized X')
    ax.set_ylabel('Normalized Y')
    ax.set_title(title)

    # Display the heatmap using Streamlit
    st.pyplot(fig)

# Display heatmap for top_right.csv
display_heatmap('top_right.csv', 'Top Right Heatmap')

# Display heatmap for bottom_left.csv
display_heatmap('bottom_left.csv', 'Bottom Left Heatmap')

display_heatmap('eye_gaze_coordinates.csv', 'Eye Gaze Heatmap')

#display scatter plot of timestamp and score in eye_gaze_coordinates.csv
df = pd.read_csv('eye_gaze_coordinates.csv')
scatter_data = df[['Timestamp', 'Score']]
fig, ax = plt.subplots()
ax.scatter(scatter_data['Timestamp'], scatter_data['Score'])
ax.set_xlabel('Timestamp')
ax.set_ylabel('Score')
ax.set_title('Score Scatter Plot')
st.pyplot(fig)

#display blinking scatter plot
df = pd.read_csv('eye_gaze_coordinates.csv')
scatter_data = df[['Timestamp', 'Blinks']]
fig, ax = plt.subplots()
ax.scatter(scatter_data['Timestamp'], scatter_data['Blinks'])
ax.set_xlabel('Timestamp')
ax.set_ylabel('Blinking Rate')
ax.set_title('Blinking Rate Scatter Plot')
st.pyplot(fig)

#get average score from csv
df = pd.read_csv('eye_gaze_coordinates.csv')
score = df['Score'].mean()
st.title("Your Study Session Productivity Score")
st.markdown(score)
if(score < 0):
  image = Image.open('dead_pokemon.webp')
  st.image(image, caption='Your pokemon died!')
else:
  image = Image.open('alive_pokemon.png')
  st.image(image, caption='Your pokemon is alive!')

client = fitbit.Fitbit(
    '23RFVG', 
    'ed6085c8a0e2a7cb173e95e1f97ab6c2',
    access_token='eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyM1JGVkciLCJzdWIiOiJCUURZOFoiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJ3aHIgd251dCB3cHJvIHdzbGUgd3dlaSB3c29jIHdzZXQgd2FjdCB3bG9jIiwiZXhwIjoxNzAwNDM0MTUzLCJpYXQiOjE3MDA0MDUzNTN9.8JhymbXAm6Yb0sM3ufyNCB5-9Go9cv-PRgYB9fAEUoc', 
    refresh_token='81d7e2d5822b4cdd585f086a1580d79c031674ea8e228bdb83c65bd26c1ed1b3'
)

start_date = arrow.get("2023-08-01")
end_date = arrow.get("2023-12-31")

# Create a series of 100-day date-range tuples between start_date and end_date
date_ranges = []
start_range = start_date
while start_range < end_date:
  if start_range.shift(days=100) < end_date:
    date_ranges.append((start_range, start_range.shift(days=100)))
    start_range = start_range.shift(days=101)
  else:
    date_ranges.append((start_range, end_date))
    start_range = end_date

# Print the result to the console
all_data = []
heart_data = []
for date_range in date_ranges:
  print(f"Requesting data for {date_range[0]} to {date_range[1]}.")
  url = f"{client.API_ENDPOINT}/1.2/user/-/sleep/date/{date_range[0].year}-{date_range[0].month:02}-{date_range[0].day:02}/{date_range[1].year}-{date_range[1].month:02}-{date_range[1].day:02}.json"
  heartrateUrl = f"{client.API_ENDPOINT}/1/user/-/activities/heart/date/{date_range[0].year}-{date_range[0].month:02}-{date_range[0].day:02}/{date_range[1].year}-{date_range[1].month:02}-{date_range[1].day:02}/15min.json"
  range_data = client.make_request(url)
  heartData = client.make_request(heartrateUrl)
  all_data.append(range_data)
  heart_data.append(heartData)
  print(f"Success!")
sleep_summaries = []
print(heart_data)

# Iterate through all data and create a list of dictionaries of results:
for data in all_data:
  for sleep in data["sleep"]:
    # For simplicity, ignoring "naps" and going for only "stage" data
    if sleep["isMainSleep"] and sleep["type"] == "stages":
      sleep_summaries.append(dict(
          date=pd.to_datetime(sleep["dateOfSleep"]).date(),
          duration_hours=sleep["duration"]/1000/60/60,
          total_sleep_minutes=sleep["minutesAsleep"],
          total_time_in_bed=sleep["timeInBed"],
          start_time=sleep["startTime"],
          deep_minutes=sleep["levels"]["summary"].get("deep").get("minutes"),
          light_minutes=sleep["levels"]["summary"].get("light").get("minutes"),
          rem_minutes=sleep["levels"]["summary"].get("rem").get("minutes"),
          wake_minutes=sleep["levels"]["summary"].get("wake").get("minutes"),            
      ))
# Convert new dictionary format to DataFrame
sleep_data = pd.DataFrame(sleep_summaries)
# Sort by date and view first rows
sleep_data.sort_values("date", inplace=True)
sleep_data.reset_index(drop=True, inplace=True)
print(sleep_data.head())
# It's useful for grouping to get the "date" from every timestamp
sleep_data["date"] = pd.to_datetime(sleep_data["date"])
# Also add a boolean column for weekend detection
sleep_data["is_weekend"] = sleep_data["date"].dt.weekday > 4
# Sleep distribution
fig, ax = plt.subplots(figsize=(12, 8))
(sleep_data["total_sleep_minutes"]/60).plot(
    kind="hist", 
    bins=50, 
    alpha=0.8,
    ax=ax
)
(sleep_data["total_time_in_bed"]/60).plot(
    kind="hist", 
    bins=50, 
    alpha=0.8
)
plt.legend()
# add some nice axis labels:
ax = plt.gca()
ax.set_xticks(range(2,12))
plt.grid("minor", linestyle=":")
plt.xlabel("Hours")
plt.ylabel("Frequency")
plt.title("Sleeping Hours")
st.pyplot(fig)
#Plot a scatter plot directly from Pandas
fig, ax = plt.subplots(figsize=(10, 10))
sleep_data.plot(
    x="total_time_in_bed", 
    y="total_sleep_minutes", 
    kind="scatter", 
    ax=ax
)
# Add a perfect 1:1 line for comparison
ax = plt.gca()
ax.set_aspect("equal")
x = np.linspace(*ax.get_xlim())
ax.plot(x,x, linestyle="--")
plt.grid(linestyle=":")
st.pyplot(fig)
# Sleep makeup - calculate data to plot
plot_data = sleep_data.\
  sort_values("date").\
  set_index("date")\
  [["deep_minutes", "light_minutes", "rem_minutes", "wake_minutes"]]
# Matplotlib doesn't natively support stacked bars, so some messing here:
df = plot_data
fig, ax = plt.subplots(figsize=(30,7), constrained_layout=True)
bottom = 0
for c in df.columns:
  ax.bar(df.index, df[c], bottom=bottom, width=1, label=c)
  bottom+=df[c]
# Set a date axis for the x-axis allows nicer tickmarks.
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
ax.legend()
plt.xlabel("Date")
plt.ylabel("Minutes")
# Show a subset of data for clarity on the website:
plt.xlim(pd.to_datetime("2023-08-01"), pd.to_datetime("2023-12-31"))
st.pyplot(fig)

# Heart Rate
heart_summaries = []
for data in heart_data:
  for heart in data["activities-heart"]:
    print(heart)
    # For simplicity, ignoring "naps" and going for only "stage" data
    if "restingHeartRate" in heart["value"] and heart["value"]["restingHeartRate"]:
      heart_summaries.append(dict(
          date=pd.to_datetime(heart["dateTime"]).date(),
          resting_heart_rate=heart["value"]["restingHeartRate"]
      ))
# Convert new dictionary format to DataFrame
heart_data = pd.DataFrame(heart_summaries)
# Sort by date and view first rows
heart_data.sort_values("date", inplace=True)
heart_data.reset_index(drop=True, inplace=True)
print(heart_data.head())
# It's useful for grouping to get the "date" from every timestamp
heart_data["date"] = pd.to_datetime(heart_data["date"])
# Also add a boolean column for weekend detection
heart_data["is_weekend"] = heart_data["date"].dt.weekday > 4
plot_data = heart_data.\
  sort_values("date").\
  set_index("date")\
  [["resting_heart_rate"]]
# Matplotlib doesn't natively support stacked bars, so some messing here:
df = plot_data
fig, ax = plt.subplots(figsize=(30,7), constrained_layout=True)
bottom = 0
for c in df.columns:
  ax.bar(df.index, df[c], bottom=bottom, width=1, label=c)
  bottom+=df[c]
# Set a date axis for the x-axis allows nicer tickmarks.
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
ax.legend()
plt.xlabel("Date")
plt.ylabel("Resting Heart Rate")
# Show a subset of data for clarity on the website:
plt.xlim(pd.to_datetime("2023-08-01"), pd.to_datetime("2023-12-31"))
st.pyplot(fig)


# sameDay = f"{client.API_ENDPOINT}/1/user/-/activities/heart/date/today/today/1min.json"
# sameDayData = client.make_request(sameDay)
# heart_summaries = []
# for heart in sameDayData["activities-heart"]:
#   # print(heart)
#   # For simplicity, ignoring "naps" and going for only "stage" data
#   if "restingHeartRate" in heart["value"] and heart["value"]["restingHeartRate"]:
#     heart_summaries.append(dict(
#         date=pd.to_datetime(heart["dateTime"]).date(),
#         resting_heart_rate=heart["value"]["restingHeartRate"]
#     ))
# Convert new dictionary format to DataFrame
# heart_data = pd.DataFrame(heart_summaries)
# # Sort by date and view first rows
# heart_data.sort_values("date", inplace=True)
# heart_data.reset_index(drop=True, inplace=True)
# print(heart_data.head())
# # It's useful for grouping to get the "date" from every timestamp
# heart_data["date"] = pd.to_datetime(heart_data["date"])
# # Also add a boolean column for weekend detection
# heart_data["is_weekend"] = heart_data["date"].dt.weekday > 4
# plot_data = heart_data.\
#   sort_values("date").\
#   set_index("date")\
#   [["resting_heart_rate"]]
# # Matplotlib doesn't natively support stacked bars, so some messing here:
# df = plot_data
# fig, ax = plt.subplots(figsize=(30,7), constrained_layout=True)
# bottom = 0
# for c in df.columns:
#   ax.bar(df.index, df[c], bottom=bottom, width=1, label=c)
#   bottom+=df[c]
# # Set a date axis for the x-axis allows nicer tickmarks.
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
# ax.legend()
# plt.xlabel("Date")
# plt.ylabel("Resting Heart Rate")
# # Show a subset of data for clarity on the website:
# plt.xlim(pd.to_datetime("2023-08-01"), pd.to_datetime("2023-12-31"))
# plt.show()

sameDay = f"{client.API_ENDPOINT}/1/user/-/activities/heart/date/today/1d/1min.json?timezone=EST"
sameDayData = client.make_request(sameDay)
# print(sameDayData["activities-heart-intraday"]["dataset"])
#graph it
heart_summaries = []
for heart in sameDayData["activities-heart-intraday"]["dataset"]:
  # print(heart)
  # For simplicity, ignoring "naps" and going for only "stage" data
  if "value" in heart:
    heart_summaries.append(dict(
        time=heart["time"],
        heart_rate=heart["value"]
    ))
# Convert new dictionary format to DataFrame
heart_data = pd.DataFrame(heart_summaries)
# Sort by date and view first rows
heart_data.sort_values("time", inplace=True)
heart_data.reset_index(drop=True, inplace=True)
print(heart_data.head())
#scatter plot
fig, ax = plt.subplots(figsize=(10, 10))
heart_data.plot(
    x="time", 
    y="heart_rate", 
    kind="scatter", 
    ax=ax
)
# make x-axis time values readable
ax = plt.gca()
ax.set_xticks(range(0,1440,60))
ax.set_xticklabels(range(0,24))
plt.xlabel("Time")
# plt.grid(linestyle=":")
# plt.show()
st.pyplot(fig)


# devices = f"{client.API_ENDPOINT}/1/user/-/devices.json"
# devicesData = client.make_request(devices)
# print(devicesData)
# deviceID = devicesData[0].get("id")

#create alarm
# alarm = f"{client.API_ENDPOINT}/1/user/-/devices/tracker/{deviceID}/alarms.json"
# time = "01:00"
# enabled = True
# weekDays = "SUNDAY"
# recurring = False
# alarm += f"?time={time}&enabled={enabled}&weekDays={weekDays}&recurring={recurring}"
# alarmData = client.make_request(alarm)
# print(alarmData)
load_dotenv()
# client = OpenAI()

# completion = client.chat.completions.create(
#   model="gpt-4-1106-preview",
#   messages=[
#     {"role": "system", "content": "You are a software helping people with productivty. Give tips for how to help people be more productive."},
#   ]
# )

# st.markdown(completion.choices[0].message.content)
# print("DONE")
client = AzureOpenAI(
  azure_endpoint = 'https://api.umgpt.umich.edu/azure-openai-api/ptu',
  api_key=os.getenv("AZURE_OPENAI_KEY"),  
  api_version="2023-03-15-preview"
)
response = client.chat.completions.create(
    model="gpt-4", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "You are a software helping people with productivty. Give tips for how to help people be more productive."},
    ]
)
st.title("Azure OpenAI Chatbot Feedback on Your Study Session Productivity")
st.markdown(response.choices[0].message.content)
productivity_data = []
with open('aw-buckets-export.json', 'r') as file:
    productivity_data = json.load(file)
eye_gazing_data = []
with open('eye_gaze_coordinates.csv', 'r') as file:
    reader = csv.reader(file)
    eye_gazing_data = list(reader)
response = client.chat.completions.create(
    model="gpt-4", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "You provide helpful productivity advice to improve study and work habits"},
        {"role": "assistant", "content": "Please provide your activity data and I will give you concise yet insightful analysis."},
        # {"role": "user", "content": f"{productivity_data}"},
        {"role": "user", "content": f"{eye_gazing_data}"}
    ]
)

st.markdown(response.choices[0].message.content)

cred_obj = firebase_admin.credentials.Certificate('alert-563fb-firebase-adminsdk-bssfd-87718064f0.json')
default_app = firebase_admin.initialize_app(cred_obj, 
    {
	'databaseURL':'https://alert-563fb-default-rtdb.firebaseio.com/',
    'storageBucket':'alert-563fb.appspot.com'
	})
fileName = "eye_gaze_coordinates.csv"
bucket = storage.bucket()
blob = bucket.blob(fileName)
blob.upload_from_filename(fileName)
