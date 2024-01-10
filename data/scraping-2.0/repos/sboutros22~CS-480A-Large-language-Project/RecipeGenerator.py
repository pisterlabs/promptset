import os
import openai
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from asyncio.windows_events import NULL

def GPT3():
    # Insert your api key here
    openai.api_key = "YOUR_API_KEY"

    # Checks if there is a food in the prompt
    if (food.get() == ''):
        return 'There is no food to generate.'
    
    prompt = 'Give me a recipe for ' + food.get() + ' it must serve ' + num_people.get() + '.'
    
    # Checks if there is any blacklisted food
    if (blacklist.get() != ''):
        prompt += '\nMust not have ' + blacklist.get() + '.'
        
    # Checks if there is any whitelisted food
    if (whitelist.get() != ''):
        prompt += '\nMust have ' + whitelist.get() + '.'
    
    
    response = openai.Completion.create(model="text-davinci-002",
                                        prompt=prompt,
                                        temperature=0.7,
                                        max_tokens=1000,
                                        top_p=1)
    return response.choices[0].text

def generate_recipe():
    """ callback when the generate button is clicked
    """
    recipe = GPT3()

    recipe_window = tk.Tk()
    recipe_window.geometry("700x800")
    recipe_window.title('Recipe')

    recipe_label = ttk.Label(recipe_window, text="Here is your custom recipe!")
    recipe_label.pack()

    recipe_text = tk.Text(recipe_window)
    recipe_text.insert('0.0', recipe)
    recipe_text.pack(fill=tk.BOTH, expand=True)

    button = ttk.Button(recipe_window, text="Click for a new recipe with the same criteria", command=generate_recipe)
    button.pack(padx=5, pady=5)

    recipe_window.mainloop()


####################################################################################################################################################################


root = tk.Tk()
root.geometry("500x250")
root.resizable(True, True)
root.title('Recipe Generator')

# store values and features
food = tk.StringVar()
num_people = tk.StringVar()
blacklist = tk.StringVar()
whitelist = tk.StringVar()


# Info frame
info = ttk.Frame(root)
info.pack(padx=10, pady=10, fill='x', expand=True)


# food
food_label = ttk.Label(info, text="What do you want me to make?")
food_label.pack(fill='x', expand=True)

food_entry = ttk.Entry(info, textvariable=food)
food_entry.pack(fill='x', expand=True)
food_entry.focus()

# Blacklisting ingredients
customization_label = ttk.Label(info, text="Would you like to blacklist any ingredients? (Comma separated list, leave blank if no)")
customization_label.pack(fill='x', expand=True)

customization_entry = ttk.Entry(info, textvariable=blacklist)
customization_entry.pack(fill='x', expand=True)

# Whitelisting ingredients
customization_label = ttk.Label(info, text="Would you like to whitelist any ingredients? (Comma separated list, leave blank if no)")
customization_label.pack(fill='x', expand=True)

customization_entry = ttk.Entry(info, textvariable=whitelist)
customization_entry.pack(fill='x', expand=True)

# Number of people
people_label = ttk.Label(info, text="How many people will this be for?")
people_label.pack(fill='x', expand=True)

people_entry = ttk.Entry(info, textvariable=num_people)
people_entry.pack(fill='x', expand=True)

# Submit button
login_button = ttk.Button(info, text="Generate Recipe", command=generate_recipe)
login_button.pack(fill='x', expand=True, pady=10)

root.mainloop()
