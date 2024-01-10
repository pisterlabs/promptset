import tkinter as tk  # Import the Tkinter library for GUI
import requests  # Import the requests library for making HTTP requests
# Import the messagebox module from Tkinter for displaying error messages
from tkinter import messagebox
# Import ImageTk and Image from PIL for working with images
from PIL import ImageTk, Image
import ttkbootstrap  # Import ttkbootstrap library for styling the Tkinter application
from io import BytesIO  # Import BytesIO from io to handle image data
import random  # Import random for theme selection
import openai


openai.api_key = "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"

# Create the Tkinter window
root = ttkbootstrap.Window(themename='morph')
root.title("Weather App")
root.geometry("500x350")
root.resizable(False, False)

# Function to fetch weather details for a given city


def get_weather(city):
    API_key = 'caa854ba9f3e141a383bca0bf29201e3'
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_key}'
    res = requests.get(url)
    if res.status_code == 404:
        messagebox.showerror('Error', 'City not found')
        return None

    # Parse the JSON response to get weather details
    weather = res.json()
    try:
        icon_id = weather['weather'][0]['icon']
        temperature = weather['main']['temp'] - 273.15
        description = weather['weather'][0]['description']
        city_name = weather['name']
        country_name = weather['sys']['country']

        # Get the icon URL and return all the weather details
        icon_url = f"https://openweathermap.org/img/wn/{icon_id}.png"
        return (icon_url, temperature, description, city_name, country_name)
    except KeyError:
        messagebox.showerror('Error', 'Failed to fetch weather details')
        return None

# Function to handle the search button click


def search():
    city = city_entry.get()
    result = get_weather(city)
    if result is None:
        return
    # If the city is found, unpack the weather information
    icon_url, temperature, description, city_name, country_name = result
    location_label.configure(text=f'{city_name}, {country_name}')
    # Get the weather icon image from the URL and update the icon label
    image_data = requests.get(icon_url, stream=True).content
    image = Image.open(BytesIO(image_data))
    icon = ImageTk.PhotoImage(image)
    icon_label.configure(image=icon)
    icon_label.image = icon
    # Update the temperature and description labels
    temp_label.configure(text=f'{temperature:.2f}°C')
    desc_label.configure(text=f'Description: {description}')

    # Generate suggestion based on weather description
    prompt = f"What to do during {description.lower()} weather?"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    suggestion = response.choices[0].text.strip()
    suggestion_label.configure(text=suggestion)


# Function to handle the theme change button click


def theme_change():
    selected_theme = random.choice(themes)
    style.theme_use(selected_theme)


# Create entry widget for city name
city_entry = ttkbootstrap.Entry(root, font='comicsans,50', width=30)
city_entry.grid(row=0, column=0, padx=20, pady=(20, 0), sticky='w')

# Create button widget for searching weather information
search_btn = ttkbootstrap.Button(
    root, text='Search Weather', width=15, command=search, bootstyle='info')
search_btn.grid(row=0, column=1, pady=(20, 0), sticky='w')

# Create label widget to display city name and country name
location_label = tk.Label(root, font='comicsans,50')
location_label.grid(row=1, column=0, columnspan=2, pady=(20, 0), sticky='n')

# Create label widget to display weather icon
icon_label = tk.Label(root)
icon_label.grid(row=2, column=0, columnspan=2, pady=20, sticky='n')

# Create label widget to display temperature
temp_label = tk.Label(root)
temp_label.config(font='comicsans,50')
temp_label.grid(row=3, column=0, columnspan=2, pady=20, sticky='n')

# Create label widget to display weather description
desc_label = tk.Label(root)
desc_label.config(font='comicsans,50')
desc_label.grid(row=4, column=0, columnspan=2, pady=20, sticky='n')

# Create an instance of ttkbootstrap.Style
style = ttkbootstrap.Style()

# Get the available themes
themes = style.theme_names()

# Create a frame for the theme button
theme_frame = ttkbootstrap.Frame(root)
theme_frame.place(relx=1, rely=1, anchor='se', bordermode='outside')

# Create label widget to display weather suggestion
suggestion_label = tk.Label(root, font='comicsans,50', wraplength=400)
suggestion_label.grid(row=5, column=0, columnspan=2, pady=20, sticky='n')

# Create the theme change button inside the frame
theme_button = ttkbootstrap.Button(
    theme_frame, text="Change Theme", command=theme_change, bootstyle="darkly")
theme_button.pack(padx=5, pady=5)


# Start the main event loop
root.mainloop()
