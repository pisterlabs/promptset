import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar, DateEntry
from amadeus import Client, ResponseError
from datetime import datetime
import json
import math
from SelectionManager import SelectionManager
import openai
import os
import sys
import json


if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

# application_path
api_key_path = os.path.join(application_path, 'JSON', 'API_key.json')
airports_path = os.path.join(application_path, 'JSON', 'airports.json')
transport_path = os.path.join(application_path, 'JSON', 'transport_info.json')
regions_path = os.path.join(application_path, 'JSON', 'regions.json')
pois_path = os.path.join(application_path, 'JSON', 'pois.json')
gpt_api_path = os.path.join(application_path, 'JSON', 'gpt_api.json')

# Load API keys from the JSON file
with open(api_key_path, 'r') as file:
    api_keys = json.load(file)
    API_KEY = api_keys['AMADEUS_API_KEY']
    API_SECRET = api_keys['AMADEUS_API_SECRET']

# Load airport data and regions data from JSON files
with open(airports_path, 'r') as file:
    airports_data = json.load(file)
    airports_korea = airports_data['Korea']
    airports_japan = airports_data['Japan']

with open(transport_path, 'r') as file:
    transport_data = json.load(file)

with open(regions_path, 'r') as file:
    regions_data = json.load(file)
    regions = regions_data

with open(pois_path, 'r') as file:
    pois_data = json.load(file)
    pois = pois_data

# Load API keys from the JSON file
with open(gpt_api_path, 'r') as file:
    gpt_api = json.load(file)
    GPT_API = gpt_api['GPT_API']

# Initialize Amadeus client
amadeus = Client(client_id=API_KEY, client_secret=API_SECRET)
#product = Client(client_id=PRODUCT_KEY, client_secret=PRODUCT_SECRET)
openai.api_key = GPT_API
# Convert specific IATA codes to 'TYO' or 'OSA'
def convert_iata_code(iata_code):
    if iata_code in ['HND', 'NRT']:
        return 'TYO'
    if iata_code in ['KIX', 'ITM']:
        return 'OSA'
    return iata_code

# Convert price from EUR to KRW
def convert_price(price):
    conversion_rate = 1350  # Example conversion rate
    return int(float(price) * conversion_rate)

JPY_TO_KRW_CONVERSION_RATE = 10  # This is an example rate, please use the current accurate rate

def convert_price_from_jpy_to_krw(price_jpy):
    return int(float(price_jpy) * JPY_TO_KRW_CONVERSION_RATE)

# Convert flight duration from ISO format to human-readable format
def convert_duration(duration):
    try:
        hours = int(duration[2:duration.find('H')])
        minutes = int(duration[duration.find('H')+1:duration.find('M')])
        return f"{hours}h {minutes}m"
    except ValueError:
        return duration

def on_mousewheel(event, canvas):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

def wrap_text(text, length):
    if len(text) > length:
        return text[:length] + '\n' + text[length:]
    else:
        return text

def update_selection_window():
    # Update Selection Information Window
    selection_window = tk.Toplevel(root)
    selection_window.title("Latest Selections")
    selection_window.geometry("300x200")

    # Displays recently selected information
    if SelectionManager.selected_flight:
        tk.Label(selection_window, text=f"Flight: {SelectionManager.selected_flight}").pack()
    if SelectionManager.selected_hotel:
        tk.Label(selection_window, text=f"Hotel: {SelectionManager.selected_hotel}").pack()
    if SelectionManager.selected_poi:
        tk.Label(selection_window, text=f"POI: {SelectionManager.selected_poi}").pack()
    if SelectionManager.selected_transport:
        tk.Label(selection_window, text=f"Transport: {SelectionManager.selected_transport}").pack()

def handle_flight_click(flight_offer):
    # Assuming flight_offer is a dictionary with the required information
    flight_details = {
        'destination': flight_offer['itineraries'][0]['segments'][-1]['arrival']['iataCode'],
        'departure_time': flight_offer['itineraries'][0]['segments'][0]['departure']['at'].split('T')[0],
        'arrival_time': flight_offer['itineraries'][1]['segments'][-1]['arrival']['at'].split('T')[0],
        'price': convert_price(flight_offer['price']['total'])
    }
    selection_manager.update_flight(flight_details)

opened_windows = []

def handle_hotel_click(hotel_offer):
    hotel_details = f"Hotel: {hotel_offer['hotel']['name']}, " \
                    f"Price: {convert_price_from_jpy_to_krw(hotel_offer['offers'][0]['price']['total'])} KRW"
    selection_manager.update_hotel(hotel_details)

def handle_poi_click(poi):
    poi_details = f"POI: {poi['Name']} - {poi['Category']} ({poi['Location']})"
    selection_manager.add_poi(poi_details)

def handle_transport_click(pass_name=None, duration=None):
    selection_manager.update_transport(pass_name=pass_name, duration=duration)

# Display flight details in a new window
def display_flight_details(flight_data):
    new_window = tk.Toplevel(root)
    opened_windows.append(new_window)
    new_window.title("Flight Details")
    new_window.geometry("1020x280+550+10")

    canvas = tk.Canvas(new_window)
    scrollbar = ttk.Scrollbar(new_window, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    if not flight_data:
        label = ttk.Label(scrollable_frame, text="Not found available Flight", font=('Gothic', 16))
        label.pack(side='top', fill='x', pady=20)
        return
    # Add the scrollbar to the canvas
    scrollbar.pack(side="right", fill="y")
    # Bind the mousewheel scrolling to the scrollbar
    new_window.bind("<MouseWheel>", lambda e: on_mousewheel(e, canvas))

    for offer in flight_data:
        round_trip_frame = ttk.Frame(scrollable_frame, padding=10, borderwidth=2, relief="groove")
        round_trip_frame.pack(fill='x', expand=True, pady=10)
        select_button = tk.Button(round_trip_frame, text="Select", command=lambda offer=offer: handle_flight_click(offer))
        select_button.pack(side='right')
        for i, itinerary in enumerate(offer['itineraries']):
            flight_info = f"Flight {i+1} - Departure: {itinerary['segments'][0]['departure']['iataCode']} {itinerary['segments'][0]['departure']['at']}, " \
                          f"Arrival: {itinerary['segments'][-1]['arrival']['iataCode']} {itinerary['segments'][-1]['arrival']['at']}, " \
                          f"Duration: {convert_duration(itinerary['duration'])}, " \
                          f"Airline: {itinerary['segments'][0]['carrierCode']}"
            label = tk.Label(round_trip_frame, text=flight_info, bg="#f0f0f0", fg="black", font=('Gothic', 12))
            label.pack(fill='x', expand=True, pady=5)

        price_info = f"Price: {convert_price(offer['price']['total'])} KRW"
        price_label = tk.Label(round_trip_frame, text=price_info, bg="#d0e0d0", fg="black", font=('Gothic', 14, 'bold'))
        price_label.pack(fill='x', expand=True, pady=5)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

# Search flights function
def search_flights():
    origin_city = origin_var.get()
    destination_city = destination_var.get()

    origin_code = next((code for code, city in airports_korea.items() if city == origin_city), None)
    destination_code = next((code for code, city in airports_japan.items() if city == destination_city), None)
    departure_date = datetime.strptime(departure_calendar.get_date(), '%m/%d/%y').strftime('%Y-%m-%d')
    return_date = datetime.strptime(return_calendar.get_date(), '%m/%d/%y').strftime('%Y-%m-%d')

    if(departure_date > return_date):
        messagebox.showerror("Error", "Departure date must be before return date")
        return

    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin_code,
            destinationLocationCode=destination_code,
            departureDate=departure_date,
            returnDate=return_date,
            adults=adults_var.get()
        )
        display_flight_details(response.data)
    except ResponseError as error:
        messagebox.showerror("Error", f"An error occurred: {error}")

# Fetch hotel list function
def fetch_hotel_list():
    destination_city = destination_var.get()
    destination_code = convert_iata_code(next((code for code, city in airports_japan.items() if city == destination_city), None))

    check_in_date = datetime.strptime(departure_calendar.get_date(), '%m/%d/%y').strftime('%Y-%m-%d')
    check_out_date = datetime.strptime(return_calendar.get_date(), '%m/%d/%y').strftime('%Y-%m-%d')

    if(check_in_date > check_out_date):
        return

    try:
        hotel_list_response = amadeus.reference_data.locations.hotels.by_city.get(cityCode=destination_code)
        hotel_ids = [hotel['hotelId'] for hotel in hotel_list_response.data]

        hotel_offers_response = amadeus.shopping.hotel_offers_search.get(
            hotelIds=hotel_ids,
            checkInDate=check_in_date,
            checkOutDate=check_out_date,
            adults=adults_var.get()
        )
        hotels = hotel_offers_response.data
        display_hotels(hotels, destination_code)
    except ResponseError as error:
        messagebox.showerror("Error", "Failed to fetch hotels: " + str(error))

def display_hotels(hotels, iata_code):
    new_window = tk.Toplevel(root)
    opened_windows.append(new_window)
    new_window.title("Hotels List")
    new_window.geometry("580x340+550+325")

    # Main frame for list of hotels
    hotels_frame = ttk.Frame(new_window)
    hotels_frame.pack(fill=tk.BOTH, expand=True)

    # Scrollable list for hotels
    hotels_canvas = tk.Canvas(hotels_frame)
    hotels_scrollbar = ttk.Scrollbar(hotels_frame, orient="vertical", command=hotels_canvas.yview)
    hotels_scrollable_frame = ttk.Frame(hotels_canvas)

    hotels_canvas.configure(yscrollcommand=hotels_scrollbar.set)
    hotels_canvas.bind('<Configure>', lambda e: hotels_canvas.configure(scrollregion=hotels_canvas.bbox("all")))
    hotels_canvas_window = hotels_canvas.create_window((0, 0), window=hotels_scrollable_frame, anchor="nw")

    if not hotels:
        label = ttk.Label(hotels_frame, text="Not found available Hotel", font=('Gothic', 16))
        label.pack(side='top', fill='x', pady=20)
        return
    
    for hotel in hotels:
        # Convert hotel prices from Yen to Korean Won.
        price_krw = convert_price_from_jpy_to_krw(hotel['offers'][0]['price']['total'])

        hotel_frame = ttk.Frame(hotels_scrollable_frame, padding=10, borderwidth=2, relief="groove")
        hotel_frame.pack(fill='x', expand=True, pady=5)
        
        hotel_name_label = ttk.Label(hotel_frame, text=hotel['hotel']['name'], font=('Gothic', 12, 'bold'))
        hotel_name_label.pack(side='left', fill='x', expand=True)
        
        hotel_price_label = ttk.Label(hotel_frame, text=f"Price: {price_krw} KRW", font=('Gothic', 12))
        hotel_price_label.pack(side='left', fill='x', expand=True)

        select_button = tk.Button(hotel_frame, text="Select", command=lambda hotel=hotel: handle_hotel_click(hotel))
        select_button.pack(side='right')

    hotels_scrollbar.pack(side='right', fill='y')
    hotels_canvas.pack(side='left', fill='both', expand=True)

    region_buttons_frame = ttk.Frame(new_window)
    region_buttons_frame.pack(fill=tk.X, expand=False)

    # Calculate number of rows needed for buttons (5 buttons per row)
    num_buttons = len(regions[iata_code].keys())
    rows_needed = (num_buttons // 5) + (num_buttons % 5 > 0)
    buttons_per_row = 5

    # Create buttons in a grid with max 5 buttons per row
    for i, (region_name, coords) in enumerate(regions[iata_code].items()):
        row = i // buttons_per_row
        column = i % buttons_per_row
        region_button = ttk.Button(region_buttons_frame, text=region_name, command=lambda rn=region_name: sort_hotels(hotels, rn, hotels_scrollable_frame, hotels_canvas, hotels_canvas_window))
        region_button.grid(row=row, column=column, padx=5, pady=5)

    # Bind the scroll event to the canvas
    new_window.bind("<MouseWheel>", lambda e: hotels_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

# Sort hotels function
def sort_hotels(hotels, region_name, hotels_scrollable_frame, hotels_canvas, hotels_canvas_window):
    destination_code = convert_iata_code(next((code for code, city in airports_japan.items() if city == destination_var.get()), None))
    region_coords = regions[destination_code][region_name]

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def get_hotel_coords(hotel):
        if 'latitude' in hotel['hotel'] and 'longitude' in hotel['hotel']:
            return hotel['hotel']['latitude'], hotel['hotel']['longitude']
        else:
            return None, None

    sorted_hotels = []
    for hotel in hotels:
        lat, lon = get_hotel_coords(hotel)
        if lat is not None and lon is not None:
            hotel['distance'] = haversine(region_coords['lat'], region_coords['lon'], lat, lon)
            sorted_hotels.append(hotel)
    sorted_hotels.sort(key=lambda x: x['distance'])

    # Clear the current hotel frames
    for widget in hotels_scrollable_frame.winfo_children():
        widget.destroy()

    # Add sorted hotel frames back into the canvas with converted prices and 'Select' buttons
    for hotel in sorted_hotels:
        price_krw = convert_price_from_jpy_to_krw(hotel['offers'][0]['price']['total'])

        hotel_frame = ttk.Frame(hotels_scrollable_frame, padding=10, borderwidth=2, relief="groove")
        hotel_frame.pack(fill='x', expand=True, pady=5)
        
        hotel_name_label = ttk.Label(hotel_frame, text=hotel['hotel']['name'], font=('Gothic', 12, 'bold'))
        hotel_name_label.pack(side='left', fill='x', expand=True)
        
        hotel_price_label = ttk.Label(hotel_frame, text=f"Price: {price_krw} KRW", font=('Gothic', 12))
        hotel_price_label.pack(side='left', fill='x', expand=True)
        
        # Add the 'Select' button for each sorted hotel
        select_button = tk.Button(hotel_frame, text="Select", command=lambda hotel=hotel: handle_hotel_click(hotel))
        select_button.pack(side='right')

    # Update the scroll region of the canvas
    hotels_canvas.configure(scrollregion=hotels_canvas.bbox("all"))

def fetch_pois(destination_city):
    departure_date = datetime.strptime(departure_calendar.get_date(), '%m/%d/%y').strftime('%Y-%m-%d')
    return_date = datetime.strptime(return_calendar.get_date(), '%m/%d/%y').strftime('%Y-%m-%d')

    if(departure_date > return_date):
        return
    # Extract the IATA code from the city name (assuming the format "City(IATA)")
    destination_city = destination_var.get()
    iata_code = convert_iata_code(next((code for code, city in airports_japan.items() if city == destination_city), None))

    pois_for_destination = pois.get(iata_code, [])
    display_pois(pois_for_destination, iata_code)


def display_pois(pois_data, iata_code):
    new_window = tk.Toplevel(root)
    opened_windows.append(new_window)
    new_window.title(f"Points of Interest in {iata_code}")
    new_window.geometry("600x340+1140+325")

    pois_canvas = tk.Canvas(new_window)
    pois_scrollbar = ttk.Scrollbar(new_window, orient="vertical", command=pois_canvas.yview)
    pois_scrollable_frame = ttk.Frame(pois_canvas)

    pois_canvas.configure(yscrollcommand=pois_scrollbar.set)
    pois_canvas.bind('<Configure>', lambda e: pois_canvas.configure(scrollregion=pois_canvas.bbox("all")))
    pois_canvas.create_window((0, 0), window=pois_scrollable_frame, anchor="nw")

    pois_scrollbar.pack(side="right", fill="y")
    pois_canvas.pack(side="left", fill="both", expand=True)

    new_window.bind("<MouseWheel>", lambda e: on_mousewheel(e, pois_canvas))

    for poi in pois_data:
        poi_frame = ttk.Frame(pois_scrollable_frame, padding=10)
        poi_frame.pack(fill='x', expand=True)

        poi_name_label = ttk.Label(poi_frame, text=poi['Name'], font=('Gothic', 12, 'bold'))
        poi_name_label.pack(side='top', fill='x', expand=True)

        poi_category_label = ttk.Label(poi_frame, text=f"Category: {poi['Category']}", font=('Gothic', 12))
        poi_category_label.pack(side='top', fill='x', expand=True)

        poi_location_label = ttk.Label(poi_frame, text=f"Location: {poi['Location']}", font=('Gothic', 12))
        poi_location_label.pack(side='top', fill='x', expand=True)

        select_button = tk.Button(poi_frame, text="Select", command=lambda poi=poi: handle_poi_click(poi))
        select_button.pack(side='right')

        # Wrap the text for descriptions longer than 35 characters
        wrapped_description = wrap_text(poi['Description'], 35)
        poi_description_label = ttk.Label(poi_frame, text=f"Description: {wrapped_description}", font=('Gothic', 12))
        poi_description_label.pack(side='top', fill='x', expand=True)

def display_transport_info(destination_code):
    departure_date = datetime.strptime(departure_calendar.get_date(), '%m/%d/%y').strftime('%Y-%m-%d')
    return_date = datetime.strptime(return_calendar.get_date(), '%m/%d/%y').strftime('%Y-%m-%d')

    if(departure_date > return_date):
        return
    destination_city = destination_var.get()
    destination_code = convert_iata_code(next((code for code, city in airports_japan.items() if city == destination_city), None))
    iata_code = convert_iata_code(destination_code)
    
    # Fetch the transport information for the IATA code
    transport_info = transport_data.get(iata_code)
    
    if transport_info:
        # Display transport information in a new window
        display_transport_details(transport_info)
    else:
        messagebox.showwarning("Not Found", "No transport information found for the selected destination.")

def display_transport_details(transport_info):
    # Create a new window to display transport details
    transport_window = tk.Toplevel(root)
    opened_windows.append(transport_window)
    transport_window.title("Transport Information")
    transport_window.geometry("+550+700")
    
    # Grid configuration for uniform button sizes
    for idx in range(1, 5):
        transport_window.grid_columnconfigure(idx, weight=1, uniform="btn_col")

    # Create a frame to hold the information
    info_frame = ttk.Frame(transport_window, padding="10")
    info_frame.pack(fill="both", expand=True)

    tk.Label(info_frame, text="Pass Name:", font=('Gothic', 12, 'bold')).grid(row=0, column=0, sticky="ew")
    tk.Label(info_frame, text="Coverage:", font=('Gothic', 12)).grid(row=1, column=0, sticky="ew")
    tk.Label(info_frame, text="Duration Options:", font=('Gothic', 12)).grid(row=2, column=0, sticky="ew")
    tk.Label(info_frame, text="Notes:", font=('Gothic', 12)).grid(row=3, column=0, sticky="ew")

    # Create buttons for each Pass Name
    pass_names = transport_info['PassName']  # Assuming this is now a list
    # Create PassName Button
    for idx, name in enumerate(transport_info['PassName']):
        btn = ttk.Button(info_frame, text=name, command=lambda n=name: handle_transport_click(pass_name=n))
        btn.grid(row=0, column=idx + 1, sticky="ew", padx=5, pady=5)

    # Coverage information
    coverage_label = tk.Label(info_frame, text=transport_info['Coverage'], font=('Gothic', 12))
    coverage_label.grid(row=1, column=1, columnspan=len(pass_names), sticky="ew")

    # Create Duration button
    duration_frame = ttk.Frame(info_frame)
    duration_frame.grid(row=2, column=1, columnspan=4, sticky="ew")
    for idx, duration in enumerate(transport_info['DurationOptions']):
        btn = ttk.Button(duration_frame, text=duration, command=lambda d=duration: handle_transport_click(duration=d))
        btn.pack(side='left', fill='x', expand=True, padx=5, pady=5)
    # Notes information
    notes_label = tk.Label(info_frame, text=transport_info['Notes'], font=('Gothic', 12))
    notes_label.grid(row=3, column=1, columnspan=len(pass_names), sticky="ew")

    # Resize window to fit content
    transport_window.update_idletasks()
    transport_window.geometry(f"{transport_window.winfo_reqwidth()}x{transport_window.winfo_reqheight()}")

def close_all_windows():
    for window in opened_windows:
        window.destroy()
    opened_windows.clear()
    selection_manager.reset_selections()

# Initialize the main GUI window
root = tk.Tk()
root.title("JTPH (Japan Travel Planning Helper)")

# Define variables for the GUI
origin_var = tk.StringVar(value='Seoul(ICN)')  # Default value
destination_var = tk.StringVar(value='Tokyo(HND)')  # Default value
adults_var = tk.IntVar(value=1)  # Default value

departure_calendar = Calendar(root, selectmode='day')
departure_calendar.grid(row=2, column=1, padx=5, pady=5)
tk.Label(root, text="Departure Date:").grid(row=2, column=0, sticky='e')

return_calendar = Calendar(root, selectmode='day')
return_calendar.grid(row=3, column=1, padx=5, pady=5)
tk.Label(root, text="Return Date:").grid(row=3, column=0, sticky='e')

# Pass variable to SelectionManager class
selection_manager = SelectionManager(root, origin_var, destination_var, adults_var, departure_calendar, return_calendar, GPT_API)

# Create widgets for the GUI
tk.Label(root, text="Origin:").grid(row=0, column=0, sticky='e')
origin_combobox = ttk.Combobox(root, textvariable=origin_var, values=list(airports_korea.values()))
origin_combobox.grid(row=0, column=1, padx=5, pady=5)

# Create widgets for the GUI
tk.Label(root, text="Origin:").grid(row=0, column=0, sticky='e')
origin_combobox = ttk.Combobox(root, textvariable=origin_var, values=list(airports_korea.values()))
origin_combobox.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Destination:").grid(row=1, column=0, sticky='e')
destination_combobox = ttk.Combobox(root, textvariable=destination_var, values=list(airports_japan.values()))
destination_combobox.grid(row=1, column=1, padx=5, pady=5)

departure_calendar = Calendar(root, selectmode='day')
departure_calendar.grid(row=2, column=1, padx=5, pady=5)
tk.Label(root, text="Departure Date:").grid(row=2, column=0, sticky='e')

return_calendar = Calendar(root, selectmode='day')
return_calendar.grid(row=3, column=1, padx=5, pady=5)
tk.Label(root, text="Return Date:").grid(row=3, column=0, sticky='e')

tk.Label(root, text="Number of Adults:").grid(row=4, column=0, sticky='e')
adults_spinbox = ttk.Spinbox(root, from_=1, to=10, textvariable=adults_var)
adults_spinbox.grid(row=4, column=1, padx=5, pady=5)

# Connect the display_transport_info to the button's command in the GUI
search_button = tk.Button(root, text="Search", command=lambda: [search_flights(), fetch_hotel_list(), fetch_pois(destination_var.get()), display_transport_info(destination_var.get())])
search_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

# Add button to close all windows
close_all_btn = tk.Button(root, text="Close All Windows", command=close_all_windows)
close_all_btn.grid(row=10, column=0, columnspan=2, pady=5)

# Start the main loop
root.mainloop()
