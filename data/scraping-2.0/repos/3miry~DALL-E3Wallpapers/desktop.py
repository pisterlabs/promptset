
import requests
import openai  # You'll need to install this package
import ctypes
import datetime
import urllib.request
import os
import winreg as reg
from openai import OpenAI
import tkinter as tk

client=OpenAI(
    api_key="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"
)

# Function to fetch the top BBC News headline using the News API
def fetch_bbc_headline(api_key):
    print("Attempting to retrieve headline")
    url = "https://newsapi.org/v2/top-headlines?sources=bbc-news"
    headers = {"Authorization": api_key}
    
    response = requests.get(url, headers=headers)
    headline = response.json()['articles'][0]['title']
    
    return headline

def fetch_weather_forecast(api_key, location):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days=1"
    response = requests.get(url)
    data = response.json()
    weather_info = data['current']['condition']['text']
    sunrise = data['forecast']['forecastday'][0]['astro']['sunrise']
    sunset = data['forecast']['forecastday'][0]['astro']['sunset']
    return weather_info, sunrise, sunset

def get_season():
    month = datetime.datetime.now().month
    if 3 <= month <= 5:
        return "Spring"
    elif 6 <= month <= 8:
        return "Summer"
    elif 9 <= month <= 11:
        return "Autumn"
    else:
        return "Winter"


def get_time_of_day(sunrise, sunset):
    now = datetime.datetime.now()
    sunrise_time = datetime.datetime.combine(now.date(), datetime.datetime.strptime(sunrise, '%I:%M %p').time())
    sunset_time = datetime.datetime.combine(now.date(), datetime.datetime.strptime(sunset, '%I:%M %p').time())
    noon = datetime.datetime.combine(now.date(), datetime.time(12, 0))

    if sunrise_time <= now < sunrise_time + datetime.timedelta(hours=2):
        return "Dawn"
    elif sunrise_time + datetime.timedelta(hours=2) <= now < noon:
        return "Morning"
    elif noon <= now < sunset_time - datetime.timedelta(hours=2):
        return "Afternoon"
    elif sunset_time - datetime.timedelta(hours=2) <= now < sunset_time:
        return "Sunset"
    else:
        return "Night"


# Function to generate an image using DALL-E
def generate_dalle_image(prompt):
    print(f"Attempting to generate image with prompt: {prompt}")
    response = client.images.generate(
        model="dall-e-3",
       prompt=prompt,
       quality="hd",
       n=1,
       size="1792x1024"
    )
    image_url = response.data[0].url
    
    return image_url

# Function to set the desktop wallpaper
def set_desktop_wallpaper(image_url):
    # Download the image
    image_path = r"C:\Users\ebevi\Pictures\AIwallpapers\wallpaper.png"
    urllib.request.urlretrieve(image_url, image_path)
    
  # Set registry for wallpaper style (Fill)
    key = reg.OpenKey(reg.HKEY_CURRENT_USER,
                      "Control Panel\\Desktop",
                      0, reg.KEY_WRITE)
    reg.SetValueEx(key, "WallpaperStyle", 0, reg.REG_SZ, "10")  # 10 is for Fill
    reg.SetValueEx(key, "TileWallpaper", 0, reg.REG_SZ, "0")
    reg.CloseKey(key)

    # Set the wallpaper (Windows)
    ctypes.windll.user32.SystemParametersInfoW(20, 0, image_path, 3)

    # Refresh desktop
    os.system("RUNDLL32.EXE USER32.DLL,UpdatePerUserSystemParameters ,1 ,True")


def get_current_location(ipinfo_api_key):
    response = requests.get(f'https://ipinfo.io?token={ipinfo_api_key}')
    data = response.json()
    print(f"Current location: {data.get('city', 'Middlesbrough')}")
    return data.get('city', 'Middlesbrough')  

def on_news_clicked(api_key):
    headline = fetch_bbc_headline(api_key)
    print(f"Fetched headline: {headline}")
    process_news_headline(headline)

def on_weather_clicked(api_key, location):
    weather_info, sunrise, sunset = fetch_weather_forecast(api_key, location)
    print(f"Fetched forecast: ({weather_info}, {sunrise}, {sunset})")
    process_weather_headline(location, weather_info, sunrise, sunset)

def process_news_headline(headline):
    image_url = generate_dalle_image(headline)
    print(f"Generated image URL: {image_url}")
    set_desktop_wallpaper(image_url)

def process_weather_headline(location, weather_info, sunrise, sunset):
    time_of_day = get_time_of_day(sunrise, sunset)
    season = get_season()
    updated_headline = f"The weather in {location} is {weather_info}. It is {time_of_day} and {season}."
    image_url = generate_dalle_image(updated_headline)
    print(f"Generated image URL: {image_url}")
    set_desktop_wallpaper(image_url)


def main():
    # Your API keys
    news_api_key = "" 
    weather_api_key = ""
    dalle_api_key = ""
    ipinfo_api_key = ''
    
    # Explicitly set the DALL-E API key
    openai.api_key = dalle_api_key

    # Fetch the current location
    current_location = get_current_location(ipinfo_api_key)

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Headline Wallpaper Generator")

    # Configure and pack the News button
    news_button = tk.Button(root, text="Fetch News Headline", command=lambda: on_news_clicked(news_api_key))
    news_button.pack()

    # Configure and pack the Weather button
    weather_button = tk.Button(root, text="Fetch Weather Forecast", command=lambda: on_weather_clicked(weather_api_key, current_location))
    weather_button.pack()

    # Start the Tkinter event loop
    root.mainloop()

# Run the main function
main()
