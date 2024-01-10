# @mansourshebli

# Import necessary libraries
from tkinter import *
from tkinter import messagebox
import openai
import os
import sys
from tkcalendar import DateEntry

# Set up OpenAI API key
try:
    openai.api_key = os.environ['OPENAI_API_KEY']
except KeyError:
    # Display a message if the API key is missing
    sys.stderr.write("""
    You haven't set up your API key yet.
    
    If you don't have an API key yet, visit:
    
    https://platform.openai.com/signup

    1. Make an account or sign in
    2. Click "View API Keys" from the top right menu.
    3. Click "Create new secret key"

    Then, open the Secrets Tool and add OPENAI_API_KEY as a secret.
    """)
    exit(1)

# Initialize variables for UI elements
interests_tk = None
budget_tk = None
departure_date_cal = None
duration_of_stay_n = None
departure_date_cal = None
return_date_cal = None
travel_companions_var = None
preferred_climate_var = None

# Function to clear text fields and reset dropdown selections
def clear_tf():
    global interests_tk, budget_tk, duration_of_stay_n, departure_date_cal, return_date_cal, travel_companions_var, preferred_climate_var

    interests_tk.delete(0, 'end')
    budget_tk.delete(0,'end')
    duration_of_stay_n.delete(0,'end')
    departure_date_cal.delete(0, 'end')
    return_date_cal.delete(0,'end')
    travel_companions_var.set('Select')
    preferred_climate_var.set('Select')

# Function to generate travel recommendations
def generate_recommendation():
    global interests_tk, budget_tk, duration_of_stay_n, departure_date_cal, return_date_cal, travel_companions_var, preferred_climate_var

    # Get values from UI elements
    interests = interests_tk.get()
    budget = budget_tk.get()
    departure_date = departure_date_cal.get()
    duration_of_stay = duration_of_stay_n.get()
    return_date = return_date_cal.get()
    travel_companions = travel_companions_var.get()
    preferred_climate = preferred_climate_var.get()

    # Create user message for the AI
    user_message = f"My interests are {interests}. My budget is {budget}. I plan to travel for {duration_of_stay} days. Return date is {return_date}. I'm traveling with {travel_companions}. I prefer a {preferred_climate} climate."

    # Define conversation messages for AI completion
    messages = [
        {"role": "system", "content": "I'm helping you plan your travel schedule based on your interests and budget."},
        {"role": "user", "content": user_message}
    ]

    # Generate AI response
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    content = response['choices'][0]['message']['content'].strip()

    # Create a result window
    result_window = Toplevel()
    result_window.title('Travel Schedule')
    result_window.config(bg='#FF0000')

    # Set up result window geometry and non-resizability
    window_width = 600
    window_height = 400
    screen_width = result_window.winfo_screenwidth()
    screen_height = result_window.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    result_window.geometry(f'{window_width}x{window_height}+{x}+{y}')
    result_window.resizable(False, False)

    # Create scrollable text widget for displaying results
    text_widget = Text(result_window, wrap=WORD)
    text_widget.pack(fill=BOTH, expand=True)

    # Process AI-generated schedule lines
    schedule_lines = content.split('\n')
    for line in schedule_lines:
        if line.startswith("Day"):
            text_widget.insert(INSERT, line + "\n", 'day_header')
        elif line.startswith(" - "):
            text_widget.insert(INSERT, line + "\n", 'activity')
        else:
            text_widget.insert(INSERT, line + "\n", 'normal')

    # Configure text widget styles
    text_widget.tag_configure('day_header', foreground='blue', font=('Helvetica', 14, 'bold'))
    text_widget.tag_configure('activity', foreground='green')
    text_widget.tag_configure('normal', font=('Helvetica', 12))

    # Add scrollbar to the text widget
    scroll_bar = Scrollbar(result_window)
    scroll_bar.pack(side=RIGHT, fill=Y)
    text_widget.config(yscrollcommand=scroll_bar.set)
    scroll_bar.config(command=text_widget.yview)
    text_widget.configure(state=DISABLED)

    # Make the result window non-resizable
    result_window.resizable(False, False)

# Create the main window for travel recommendations
window = Tk()
window.title('Travel Destination Advisor')
window.geometry('800x500')
window.config(bg='#00FFFF')

var = IntVar()  # Initialize an IntVar for radio buttons

# Create the main frame
frame = Frame(window, padx=100, pady=100, bg='#00FFFF')
frame.grid(row=0, column=0, sticky='nsew')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# Create and position UI elements


title_label = Label(frame, text='Destination Recommendations', font=('Helvetica', 24, 'bold', 'italic'), bg='#00FFFF')
title_label.grid(row=0, column=1, columnspan=2, sticky="w")


interests_lb = Label(frame, text="What are some of your interests? (Separate each by ',')", font=('Arial', 12), bg='#00FFFF')
interests_lb.grid(row=1, column=1, sticky="w")

interests_tk = Entry(frame)
interests_tk.grid(row=1, column=2, pady=5, sticky="w")

budget_lb = Label(frame, text="Budget$:", font=('Arial', 12), bg='#00FFFF')
budget_lb.grid(row=2, column=1, sticky="w")

budget_tk = Entry(frame)
budget_tk.grid(row=2, column=2, pady=5, sticky="w")

duration_of_stay_lb = Label(frame, text="Duration of Stay:", font=('Arial', 12), bg='#00FFFF')
duration_of_stay_lb.grid(row=3, column=1, sticky="w")

duration_of_stay_n = Entry(frame)
duration_of_stay_n.grid(row=3, column=2, pady=5, sticky="w")

departure_date_lb = Label(frame, text="Departure Date:", font=('Arial', 12), bg='#00FFFF')
departure_date_lb.grid(row=4, column=1, sticky="w")

departure_date_cal = DateEntry(frame, date_pattern='yyyy-mm-dd')
departure_date_cal.grid(row=4, column=2, pady=5, sticky="w")

return_date_lb = Label(frame, text="Return Date:", font=('Arial', 12), bg='#00FFFF')
return_date_lb.grid(row=5, column=1, sticky="w")

return_date_cal = DateEntry(frame, date_pattern='yyyy-mm-dd')
return_date_cal.grid(row=5, column=2, pady=5, sticky="w")

travel_companions_var = StringVar()
travel_companions_var.set("Select")

travel_companions_lb = Label(frame, text="Travel Companions:", font=('Arial', 12), bg='#00FFFF')
travel_companions_lb.grid(row=6, column=1, sticky="w")

travel_companions_options = ["Select", "Solo", "Family", "Couples", "Friends"]
travel_companions_menu = OptionMenu(frame, travel_companions_var, *travel_companions_options)
travel_companions_menu.grid(row=6, column=2, pady=5, sticky="w")

preferred_climate_var = StringVar()
preferred_climate_var.set("Select")

preferred_climate_lb = Label(frame, text="Preferred Climate:", font=('Arial', 12), bg='#00FFFF')
preferred_climate_lb.grid(row=7, column=1, sticky="w")

preferred_climate_options = ["Select", "Warm and tropical", "Cool and mountainous", "Moderate and temperate"]
preferred_climate_menu = OptionMenu(frame, preferred_climate_var, *preferred_climate_options)
preferred_climate_menu.grid(row=7, column=2, pady=5, sticky="w")

# Create a sub-frame for buttons
frame2 = Frame(frame, bg='#00FFFF')
frame2.grid(row=8, columnspan=3, pady=10, sticky="w")

generate_recommendation_btn = Button(frame2, text='Generate Recommendation', command=generate_recommendation, bg="green", fg="white", font=("Arial", 12))
generate_recommendation_btn.pack(side=LEFT)

reset_btn = Button(frame2, text='Reset', command=clear_tf, bg="blue", fg="white", font=("Arial", 12))
reset_btn.pack(side=LEFT)

exit_btn = Button(frame2, text='Exit', command=window.destroy, bg="red", fg="white", font=("Arial", 12))
exit_btn.pack(side=LEFT)

# Start the main event loop
window.mainloop()
