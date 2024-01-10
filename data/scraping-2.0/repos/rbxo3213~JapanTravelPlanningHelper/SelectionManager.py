import tkinter as tk
from tkinter import ttk, messagebox

class SelectionManager:
    def __init__(self, root, origin_var, destination_var, adults_var, departure_calendar, return_calendar, gpt_api_key):
        self.root = root
        self.GPT_API = gpt_api_key
        self.selected_flight = None
        self.selected_hotel = None
        self.selected_pois = []
        self.selected_transport = None
        self.selected_pass_names = []
        self.selected_durations = []
        self.origin_var = origin_var
        self.destination_var = destination_var
        self.adults_var = adults_var
        self.departure_calendar = departure_calendar
        self.return_calendar = return_calendar

        # Create a frame to display the selection
        self.selection_frame = ttk.LabelFrame(root, text="Current Selections", padding=(20, 10))
        self.selection_frame.grid(row=6, column=0, columnspan=2, sticky="ew")

        # Create a label to display your selection
        self.flight_label = ttk.Label(self.selection_frame, text="")
        self.flight_label.pack()

        self.hotel_label = ttk.Label(self.selection_frame, text="")
        self.hotel_label.pack()

        self.pois_label = ttk.Label(self.selection_frame, text="")
        self.pois_label.pack()

        # Create and set a Transport label.
        self.transport_label = ttk.Label(self.selection_frame, text="")
        self.transport_label.pack()

        # Create a reset button
        self.reset_button = ttk.Button(self.selection_frame, text="Reset", command=self.reset_selections)
        self.reset_button.pack()

        # Create OK button and change command settings
        self.ok_button = ttk.Button(self.selection_frame, text="Make Plan", command=self.confirm_and_generate_plan)
        self.ok_button.pack()

    def update_flight(self, flight_info):
        # Assuming flight_info is a dictionary with the necessary information
        formatted_flight_details = (
            f"Flight: Destination: {flight_info['destination']}, "
            f"Price: {flight_info['price']} KRW, \n"
            f"Departure: {flight_info['departure_time']}, "
            f"Arrival: {flight_info['arrival_time']}"
        )
        self.selected_flight = formatted_flight_details
        self.flight_label.config(text=formatted_flight_details)

    def update_hotel(self, hotel_info):
        self.selected_hotel = hotel_info
        self.hotel_label.config(text=f"{hotel_info}")

    def add_poi(self, poi_info):
        # If the POI is not already selected, append it to the list
        if poi_info not in self.selected_pois:
            self.selected_pois.append(poi_info)
            # Update the POIs label with each POI on a new line
            pois_text = "\n".join(self.selected_pois)
            self.pois_label.config(text=f"{pois_text}")

    def update_transport(self, pass_name=None, duration=None):
        if pass_name:
            self.selected_pass_names = [pass_name]  # Replace list with current pass_name
        if duration:
            self.selected_durations = [duration]  # Replace list with current duration
        # Print PassName and Duration together
        self.transport_label.config(text=f"PassName: {', '.join(self.selected_pass_names)}, Duration: {', '.join(self.selected_durations)}")


    def reset_selections(self):
        self.selected_flight = None
        self.selected_hotel = None
        self.selected_pois = []
        self.selected_transport = None
        self.flight_label.config(text="")
        self.hotel_label.config(text="")
        self.pois_label.config(text="")
        self.transport_label.config(text="")

    def confirm_selections(self):
        # Selection confirmation logic
        # For example, you can save your selections to a file or pass them to another process.
        print(f"Confirmed selections: {self.selected_flight}, {self.selected_hotel}, {self.selected_pois}, {self.selected_transport}")

    def generate_travel_plan(self):
        # Construct the prompt for GPT-3
        travel_details = f"Travel to {self.destination_var.get()} from {self.origin_var.get()} for {self.adults_var.get()} adults. " \
                         f"Departure on {self.departure_calendar.get_date()} and return on {self.return_calendar.get_date()}. " \
                         f"Includes flight: {self.selected_flight}, hotel: {self.selected_hotel}, " \
                         f"POIs: {', '.join(self.selected_pois)}, " \
                         f"Transport: {', '.join(self.selected_pass_names)}, Duration: {', '.join(self.selected_durations)}. " \
                         "Please suggest a travel plan."
    
        try:
            # Updated API call
            from openai import OpenAI
            client = OpenAI(api_key=self.GPT_API)
            response = client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": travel_details
                }],
                model="gpt-3.5-turbo"
            )
            # Displaying the result in a new window
            result_window = tk.Toplevel(self.root)
            result_window.title("Travel Plan Suggestion")
            result_window.geometry("600x400")
    
            result_text = tk.Text(result_window, wrap="word")
            if response.choices and response.choices[0].message:
                result_text.insert("1.0", response.choices[0].message.content)
            result_text.pack(expand=True, fill="both")
    
        except Exception as e:  # Generic exception handling, consider specifying exact exceptions
            messagebox.showerror("Error", f"An error occurred while generating the travel plan: {e}")


    def get_selections(self):
        # Returns all information selected by the user
        return {
            'flight': self.selected_flight,
            'hotel': self.selected_hotel,
            'pois': self.selected_pois,
            'transport': {
                'pass_names': self.selected_pass_names,
                'durations': self.selected_durations
            }
        }
        
    def confirm_and_generate_plan(self):
        # Add verification logic
        if not self.selected_flight or not self.selected_hotel or not self.selected_pois or not self.selected_pass_names or not self.selected_durations:
            messagebox.showerror("Error", "Please select all options.")
            return 
        
        self.confirm_selections()
        self.generate_travel_plan()

