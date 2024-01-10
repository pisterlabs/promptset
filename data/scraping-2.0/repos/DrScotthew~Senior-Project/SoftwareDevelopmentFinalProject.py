from ast import Try, TryStar
from cgitb import lookup
from http.server import ThreadingHTTPServer
import os
from pickle import FALSE
import string
from unittest import skip
from InventoryVersion1 import Inventory_Version_1 #import inventory management inventory_version_1.py
import colorama
#Colorama is a way to highlight text color in the output console
from colorama import just_fix_windows_console
#'Fixes' windows and forces colored text to work
from colorama import init
from colorama import Fore   #ability for color change settings for objects...see 'settings'
import json
import traceback
import random
import StarWarsDataSets
import SupernaturalDataSets
import DemoDataSets
import openai
from openai import OpenAI   #uses openai
import config
import time

global isUsingOpenAI

client = OpenAI(
  organization=openai.api_key,
)

def random_generation_ask():
    print("Would you like to use the OpenAI random generation? (see README for more info...)")    #asks user if they want to use OpenAI random generation
    choice = input()
    global isUsingOpenAI
    if choice in ("yes", "Y", "y"):
        print("Do you have your own API key to use?")   #asks user if they have their own API key to use
        choice2=input()
        if choice2 in ("yes", "Y", "y"):
            print("Please type in the API key you would like to use: ")     #asks user to put in their API key to use
            isUsingOpenAI = True
            user_api_key = input()
            openai.api_key = user_api_key      #sets api key to what user put in...
            new_game_screen()
        else:
            print("The game will use preconfigured data sets...")     #will use the prechosen OpenAI key so user can use OpenAI random generation still...
            isUsingOpenAI = False
            new_game_screen()
    else:
        print("The game will use preconfigured data sets...")   #uses the preconfigured data sets...
        isUsingOpenAI=False
        new_game_screen()

random_generation_spn_weapon_names = []      #creates spn array elements for weapon names
random_generation_starwars_weapon_names = []     #creates star wars array elements for weapon names
random_generation_demo_weapon_names = []        #creates demo array elements for weapon names
random_generation_spn_weapon_descriptions = []      #creates spn array elements for weapon descriptions...array index values will match weapon array index values each time as both are added at same time...
random_generation_starwars_weapon_descriptions = []     #star wars weapon descriptions array
random_generation_demo_weapon_descriptions = []         #demo weapon descriptions array

def demo_openai_gen():
    demo_prompt = ["give the name of a random weapon that is not listed here: {}".format(random_generation_demo_weapon_names) + "\n\n"]
        #will ask for generic weapons for demo mode...each weapon must be unique...
    response = openai.completions.create(
        model="text-davinci-003",     #using davinci003
        prompt=demo_prompt,  #asks openai to give a weapon name from 'Supernatural'
        temperature=0.7,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=10
    )
    demo_weapon = response.choices[0].text;
    random_generation_demo_weapon_names.append(demo_weapon)      #adds weapon name to array...
    L = [demo_weapon]
    demo_file = open('random_weapons_demo.txt', 'w')     #saves each weapon name into new .txt file
    demo_file.writelines(L)
    demo_file.close
    
    demo_file = open('random_weapons_demo.txt', 'r')
    demo_prompt2=["give a one sentence description of " + demo_weapon + "describing what it looks like and its function.  Do not use the phrase '" + demo_weapon + "' in your description." +"\n\n"]
        #asks openai for description of demo weapon(s)...
    response = openai.completions.create(
        model="text-davinci-003",
        prompt=demo_prompt2,  #asks openai to give a description from previously used name for demo weapon
        temperature=0.7,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=50
    )
    demo_weapon_description = response.choices[0].text;
    random_generation_demo_weapon_descriptions.append(demo_weapon_description)       #adds weapon description to array...
    final_weapon_description_save = open('random_weapons_demo_descriptions.txt', 'w')
    final_weapon_description_save.writelines(demo_weapon_description)
    final_weapon_description_save.close

def supernatural_openai_gen():   #all random generation for supernatural
    supernatural_prompt = ["give a random weapon name from the tv show 'Supernatural' that is not listed here: {}".format(random_generation_spn_weapon_names) + "\n\n"]
        #asks openai to give one weapon name each time...this is to prevent errors when ai gives list of items and program is unable to accurately prepare to read txt file in different formats...
        #also asks for openai to only give a weapon name that is unique from weapon names already stored in array...
    response = openai.completions.create(
        model="text-davinci-003",     #using davinci003
        prompt=supernatural_prompt,  #asks openai to give a weapon name from 'Supernatural'
        temperature=0.7,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=10
    )
    supernatural_weapon = response.choices[0].text;
    random_generation_spn_weapon_names.append(supernatural_weapon)      #adds weapon name to array...
    L = [supernatural_weapon]
    spn_file = open('random_weapons_supernatural.txt', 'w')     #saves each weapon name into new .txt file
    spn_file.writelines(L)
    spn_file.close
    
    spn_file = open('random_weapons_supernatural.txt', 'r')
    supernatural_prompt2=["give a one sentence description of the weapon from 'Supernatural' called " + supernatural_weapon + "describing what it looks like and its function.  Do not use the phrase '" + supernatural_weapon + "' in your description." +"\n\n"]
        #this will ask the openai to give a description for the weapon stored in supernatural_weapon variable without mentioning the weapon's name...this is to make the text in the game less redundant and awkward...
        #need to check each time if program has already given a description for that weapon...? no since it will be called each time weapon is created
        #will store descriptions in save file...
        #will access said save file when player wants to see weapon descriptions from inventory screen...
            #e.g. player opens inventory, selects weapon, program displays weapon description again for player before asking if thats the weapon they want to use...
            #program will know which description to give based on array index value...since program gives description at same time as weapon name, the index values will match each time...
    response = openai.completions.create(
        model="text-davinci-003",     #using davinci003
        prompt=supernatural_prompt2,  #asks openai to give a description from previously used name from 'Supernatural'
        temperature=0.7,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=50
    )
    supernatural_weapon_description = response.choices[0].text;
    random_generation_spn_weapon_descriptions.append(supernatural_weapon_description)       #adds weapon description to array...
    final_weapon_description_save = open('random_weapons_supernatural_descriptions.txt', 'w')
    final_weapon_description_save.writelines(supernatural_weapon_description)
    final_weapon_description_save.close

def starwars_openai_gen():
    starwars_prompt = ["give a random weapon name from 'Star Wars' that is not listed here: {}".format(random_generation_starwars_weapon_names) + "\n\n"]
    response = openai.completions.create(
    model="text-davinci-003",     #using davinci003
    prompt=starwars_prompt,     #asks openai to give a weapon name from 'Star Wars'
    temperature=0.7,
    frequency_penalty=0,
    presence_penalty=0,
    max_tokens=50
    )
    starwars_weapon = response.choices[0].text;
    random_generation_starwars_weapon_names.append(starwars_weapon)
    L1 = [starwars_weapon]
    starwars_file = open('random_weapons_starwars.txt', 'w')
    starwars_file.writelines(L1)
    starwars_file.close

    starwars_file = open('random_weapons_starwars.txt', 'r')
    starwars_prompt2=["give a one sentence description of the weapon from 'Supernatural' called " + starwars_weapon + "describing what it looks like and its function.  Do not use the phrase '" + starwars_weapon + "' in your description." +"\n\n"]
    response = openai.completions.create(
        model="text-davinci-003",     #using davinci003
        prompt=starwars_prompt2,    #asks openai to give a description from previously used name from 'Supernatural'
        temperature=0.7,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=50
    )
    starwars_weapon_description = response.choices[0].text;
    random_generation_starwars_weapon_descriptions.append(starwars_weapon_description)       #adds weapon description to array...
    final_weapon_description_save = open('random_weapons_supernatural_descriptions.txt', 'w')
    final_weapon_description_save.writelines(starwars_weapon_description)
    final_weapon_description_save.close

def display_title_screen():
    os.system("clear" if os.name == "posix" else "cls")  # Clear the screen

    title = """
_______  _______          _________          _______           _______ 
(  ____ )(  ___  )|\     /|\__   __/|\     /|(  ____ )|\     /|(  ____ \\
| (    )|| (   ) || )   ( |   ) (   | )   ( || (    )|| )   ( || (    \/
| (____)|| |   | || |   | |   | |   | |   | || (____)|| |   | || (_____ 
|     __)| |   | |( (   ) )   | |   ( (   ) )|     __)| |   | |(_____  )
| (\ (   | |   | | \ \_/ /    | |    \ \_/ / | (\ (   | |   | |      ) |
| ) \ \__| (___) |  \   /  ___) (___  \   /  | ) \ \__| (___) |/\____) |
|/   \__/(_______)   \_/   \_______/   \_/   |/   \__/(_______)\_______)
"""
    menu = """
    1. Start New Game
    2. Load Saved Game
    3. Settings
    4. Exit Game
    """

    print(title)
    print(menu)

def display_new_game_screen():
    #asks player if they want to enter a custom seed or use a pregenerated random seed for their new game
    os.system("clear" if os.name == "posix" else "cls")  # Clear the screen

    open('random_weapons_supernatural.txt', 'w').close()    #clears all data
    open('random_weapons_starwars.txt', 'w').close()        #clears all data
    open('save.txt', 'w').close()

    seed_selection = """
    Enter a custom seed.  Alternatively, type 'random' for a random seed.  Enter 'exit' to go back to the main menu.
    """

    print(seed_selection)

def display_settings_screen():
    #displays all available color choices player can choose from before starting their game
    color_selection = """ Please choose a color for highlighted items.  (*Note: The default highlight color for items is Yellow.  All enemies will have a Red highlight color regardless of choice for object color.).
    
    1. Blue
    2. Yellow
    3. Purple
    4. Red
    5. Green

    Enter 'exit' to go back to the main menu.
    """

    print(color_selection)

def data_starwars():    #accesses data sets for star wars
    global place
    global weapon
    global weapon_name
    global weapon_description
    global trash_can
    global trash_can_name
    global piece_of_paper
    global piece_of_paper_name
    global enemy
    global enemy_name

    if isUsingOpenAI:
        starwars_openai_gen()      #performs openai generation for weapon names/descriptions
        weapon_name = random.choice(random_generation_starwars_weapon_names)     #will access from randomly generated lists
        weapon_description = random.choice(random_generation_starwars_weapon_descriptions)
        place = random.choice(StarWarsDataSets.woods)   #randomly chooses values from specified list
        trash_can_name = random.choice(StarWarsDataSets.trash_can)
        piece_of_paper_name = random.choice(StarWarsDataSets.piece_of_paper)
        enemy_name = random.choice(StarWarsDataSets.enemy)
    else:
        weapon_name = random.choice(StarWarsDataSets.weapons)      #will access from predefined data sets
        weapon_description = "test description of star wars weapon..."
        place = random.choice(StarWarsDataSets.woods)   #randomly chooses values from specified list
        trash_can_name = random.choice(StarWarsDataSets.trash_can)
        piece_of_paper_name = random.choice(StarWarsDataSets.piece_of_paper)
        enemy_name = random.choice(StarWarsDataSets.enemy)
    try: 
        weapon = color_choice + weapon_name + Fore.RESET
        trash_can = color_choice + trash_can_name + Fore.RESET
        piece_of_paper = color_choice + piece_of_paper_name + Fore.RESET
        enemy = Fore.RED + enemy_name + Fore.RESET
    except:
        weapon = Fore.YELLOW + weapon_name + Fore.RESET
        trash_can = Fore.YELLOW + trash_can_name + Fore.RESET
        piece_of_paper = Fore.YELLOW + piece_of_paper_name + Fore.RESET
        enemy = Fore.RED + enemy_name + Fore.RESET

def data_supernatural():    #accesses data sets for supernatural
    global place
    global weapon
    global weapon_name
    global weapon_description
    global trash_can
    global trash_can_name
    global piece_of_paper
    global piece_of_paper_name
    global enemy
    global enemy_name

    if isUsingOpenAI:
        supernatural_openai_gen()      #performs openai generation for weapon names/descriptions
        weapon_name = random.choice(random_generation_spn_weapon_names)     #will access from randomly generated lists
        weapon_description = random.choice(random_generation_spn_weapon_descriptions)
        place = random.choice(SupernaturalDataSets.woods)   #randomly chooses values from specified list
        trash_can_name = random.choice(SupernaturalDataSets.trash_can)
        piece_of_paper_name = random.choice(SupernaturalDataSets.piece_of_paper)
        enemy_name = random.choice(SupernaturalDataSets.enemy)
    else:
        weapon_name = random.choice(SupernaturalDataSets.weapons)      #will access from predefined data sets
        weapon_description = "test description of supernatural weapon..."
        place = random.choice(SupernaturalDataSets.woods)   #randomly chooses values from specified list
        trash_can_name = random.choice(SupernaturalDataSets.trash_can)
        piece_of_paper_name = random.choice(SupernaturalDataSets.piece_of_paper)
        enemy_name = random.choice(SupernaturalDataSets.enemy)
    try: 
        weapon = color_choice + weapon_name + Fore.RESET
        trash_can = color_choice + trash_can_name + Fore.RESET
        piece_of_paper = color_choice + piece_of_paper_name + Fore.RESET
        enemy = Fore.RED + enemy_name + Fore.RESET
    except:
        weapon = Fore.YELLOW + weapon_name + Fore.RESET
        trash_can = Fore.YELLOW + trash_can_name + Fore.RESET
        piece_of_paper = Fore.YELLOW + piece_of_paper_name + Fore.RESET
        enemy = Fore.RED + enemy_name + Fore.RESET

def data_demo():    #accesses data set for traditional demo
    global place
    global weapon
    global weapon_name
    global weapon_description
    global trash_can
    global trash_can_name
    global piece_of_paper
    global piece_of_paper_name
    global enemy
    global enemy_name
    if isUsingOpenAI:
        demo_openai_gen()      #performs openai generation for weapon names/descriptions
        weapon_name = random.choice(random_generation_demo_weapon_names)     #will access from randomly generated lists
        weapon_description = random.choice(random_generation_demo_weapon_descriptions)
        place = random.choice(DemoDataSets.woods)   #randomly chooses values from specified list
        trash_can_name = random.choice(DemoDataSets.trash_can)
        piece_of_paper_name = random.choice(DemoDataSets.piece_of_paper)
        enemy_name = random.choice(DemoDataSets.enemy)
    else:
        weapon_name = random.choice(DemoDataSets.weapons)      #will access from predefined data sets
        weapon_description = "test description of demo weapon..."
        place = random.choice(DemoDataSets.woods)   #randomly chooses values from specified list
        trash_can_name = random.choice(DemoDataSets.trash_can)
        piece_of_paper_name = random.choice(DemoDataSets.piece_of_paper)
        enemy_name = random.choice(DemoDataSets.enemy)
    try: 
        weapon = color_choice + weapon_name + Fore.RESET
        trash_can = color_choice + trash_can_name + Fore.RESET
        piece_of_paper = color_choice + piece_of_paper_name + Fore.RESET
        enemy = Fore.RED + enemy_name + Fore.RESET
    except:
        weapon = Fore.YELLOW + weapon_name + Fore.RESET
        trash_can = Fore.YELLOW + trash_can_name + Fore.RESET
        piece_of_paper = Fore.YELLOW + piece_of_paper_name + Fore.RESET
        enemy = Fore.RED + enemy_name + Fore.RESET

def settings_screen():
     #uses ASCI codes to change colors...
     display_settings_screen()
     global color_choice
     for color_choice in [Fore.BLUE, Fore.YELLOW, Fore.MAGENTA, Fore.RED, Fore.GREEN]:
         while True:
            choice = input("Make your choice: ")

            if choice == '1':
                data.update({'highlight_color': Fore.BLUE})
                print(Fore.BLUE + 'Highlight color changed to Blue')
                color_choice = Fore.BLUE
                print(Fore.RESET)   #resets color of text in console back to white...object color is set, however

            elif choice == '2':
                data.update({'highlight_color': Fore.YELLOW})
                print(Fore.YELLOW + 'Highlight color changed to Yellow')
                color_choice = Fore.YELLOW
                print(Fore.RESET)

            elif choice == '3':
                data.update({'highlight_color': Fore.MAGENTA})
                print(Fore.MAGENTA + 'Highlight color changed to Purple')
                color_choice = Fore.MAGENTA
                print(Fore.RESET)

            elif choice == '4':
                data.update({'highlight_color': Fore.RED})
                print(Fore.RED + 'Highlight color changed to Red')
                color_choice = Fore.RED
                print(Fore.RESET)
           
            elif choice == '5':
                data.update({'highlight_color': Fore.GREEN})
                print(Fore.GREEN + 'Highlight color changed to Green')
                color_choice = Fore.GREEN
                print(Fore.RESET)

            elif choice == 'go back':
                main()

            else:
                print("Invalid input.  Please choose a highlight color from the list.")
     


color_choice = [Fore.BLUE, Fore.YELLOW, Fore.MAGENTA , Fore.RED, Fore.GREEN]

setting = [DemoDataSets.woods, StarWarsDataSets.woods, SupernaturalDataSets.woods]

data = {
    'checkpoint': 0,
    'complete inventory': Inventory_Version_1.Inventory,
    'weapons': Inventory_Version_1.InventoryWeapons,
    'armor': Inventory_Version_1.InventoryArmor,
    'ammo': Inventory_Version_1.InventoryAmmo,
    'health items': Inventory_Version_1.InventoryHealthItems,
    'currently equipped': Inventory_Version_1.InventoryCurrentlyEquipped,
    'money': Inventory_Version_1.InventoryMoney,
    'highlight_color': Fore.YELLOW,
    'setting': DemoDataSets.woods
    #this allows for the player to load their game from a save file
    }

def main():
    #automatically goes to title screen from main() function
    main_menu()

    choice = input()
    if choice == 'exit game':
        print("Saving the game...")
    checkpoints = [start, startpath, startpath_object, startpath_continue, crossroads, crossroads_left, crossroads_left1, crossroads_left2, crossroads_left3]

def load_game():
    global color_choice
    color_choice = [Fore.BLUE, Fore.YELLOW, Fore.MAGENTA, Fore.RED, Fore.GREEN]
    checkpoints = [start, startpath, startpath_object, startpath_continue, crossroads, crossroads_left, crossroads_left1, crossroads_left2, crossroads_left3]
    #checkpoints are defined as places in the story or 'timeline' for the player to access...
    #this allows for the game to always know where the player is in the story

    try:
        with open('save.txt') as save_file:
            data = json.load(save_file)     #gets data from save file...
            Inventory_Version_1.Inventory = data.get("inventory")   #checks what items player had in save file
            Inventory_Version_1.InventoryWeapons = data.get("weapons")
            Inventory_Version_1.InventoryArmor = data.get("armor")
            Inventory_Version_1.InventoryAmmo = data.get("ammo")
            Inventory_Version_1.InventoryHealthItems = data.get("health items")
            Inventory_Version_1.InventoryCurrentlyEquipped = data.get("currently equipped")
            Inventory_Version_1.InventoryMoney = data.get("money")
            color_choice = data.get("highlight_color")
            saved_setting = data.get("setting")
            if saved_setting == 'supernatural':
                data_supernatural()
            elif saved_setting == 'star wars':
                data_starwars()
            else:
                data_demo()
            saved_checkpoint = data.get("checkpoint")   #checks where player was when they exited the game last
            checkpoints[saved_checkpoint]()     #puts player back at same 'checkpoint' they were in last
    except:
        start()

def main_menu():
    #the title screen
    display_title_screen()

    while True:
        choice = input("Make Your Choice (1/2/3/4): ")
        
        if choice == '1':
            # Allows player to either a) enter a custom seed or b) use a pregenerated random seed in order to create the world and start the game
            random_generation_ask()     #will ask player if they want to use the OpenAI random generation or preconfigured data sets...
            
        elif choice == '2':
            # Loads last saved game
            print("Loading a saved game...")
            load_game()

        elif choice == '3':
            # Allows player to change different settings (includes: highlight color)
            settings_screen()
            
        elif choice == '4':
            # Exits the game
            print("Exiting the game...")
            
        else:
            # Error control...
            print("Invalid choice. Please select a valid option (1/2/3/4).")

#def random_generation_ask():
#    print("Would you like to use the OpenAI random generation?")
#    choice = input()

#    if choice in ("yes", "Y", "y"):
#        print("Do you have your own API key to use?")
#        choice2=input()
#        if choice2 in ("yes", "Y", "y"):
#            print("Please type in the API key you would like to use: ")
#            isUsingOpenAI = True
#            open_api_key = input()      #sets api key to what user put in...
#            new_game_screen()
#        else:
#            openai.api_key = config.api_key     #sets api key to example api key...
#            isUsingOpenAI = True
#            new_game_screen()
#    else:
#        print("The game will use preconfigured data sets...")
#        new_game_screen()

def new_game_screen():
    display_new_game_screen()

    while True:
        choice = input()

        if choice == 'random': #for demo playthrough
            #Will generate a random seed for the world the player will be in...however, this is currently for the demo
            print("Generating random seed...")
            print("Loading...")
            Inventory_Version_1.isDemo=True
            data.update({'setting': 'demo'})    #updates saved selection of data set used
            Inventory_Version_1.Inventory.clear     #clears inventory data for new game
            data_demo()     #will assign demo data for wold generation
            start()     #goes to start function to start game

        elif choice == 'star wars':
            print("Generating a Star Wars themed world...")
            print("Loading...")
            Inventory_Version_1.isStarWars=True
            data.update({'setting': 'star wars'})
            Inventory_Version_1.Inventory.clear
            data_starwars()     #will assign star wars data sets for wold generation
            start()

        elif choice == 'supernatural':
            print("Generating a Supernatural themed world...")
            print("Loading...")
            Inventory_Version_1.isSupernatural=True
            data.update({'setting': 'supernatural'})
            Inventory_Version_1.Inventory.clear
            data_supernatural()     #will assign supernatural data sets for world generation
            start()
            
        elif choice == 'exit':
            main_menu()
            
        else:   #incomplete...will finish later
            print("Generating world based on seed given...")
            print("Loading...")

flowchart = []  #the flowchart for the player's choices in the game...

def weapon_inventory_descriptions():
    stack = traceback.extract_stack()
    filename, codeline, funcName, text = stack[-2]
    checkpoints = {'start': start, 'startpath':startpath, "startpath_object":startpath_object, "startpath_continue":startpath_continue, "crossroads":crossroads, "crossroads_left":crossroads_left, "crossroads_left1":crossroads_left1, "crossroads_left2":crossroads_left2}
    last_checkpoint = flowchart[-1]     #gets last checkpoint in player's flowchart
    if not Inventory_Version_1.InventoryWeapons:    #checks if player has weapons...if not, tells them they don't have any weapons and takes them back...
        checkpoints[last_checkpoint]()  #returns player to previous checkpoint...
    else:  
        print("Which weapon do you want to look at?")
        while True:
            choice2 = input()
            if choice2=="{}".format(Inventory_Version_1.weapon_description_index) and Inventory_Version_1.isStarWars:      #and Inventory_Version_1.isStarWars
                print("Description: ")      #how to make program look at descriptions for weapon...how to make it know if using spn data sets or star wars data sets...
                #print(random_generation_starwars_weapon_descriptions[Inventory_Version_1.weapon_description_index])
                print(weapon_description)
                checkpoints[last_checkpoint]()      #returns player to last checkpoint...
            elif choice2=="{}".format(Inventory_Version_1.weapon_description_index) and Inventory_Version_1.isSupernatural:
                print("Description: ")      #how to make program look at descriptions for weapon...how to make it know if using spn data sets or star wars data sets...
                print(random_generation_spn_weapon_descriptions[Inventory_Version_1.weapon_description_index])
                checkpoints[last_checkpoint]()      #returns player to last checkpoint...
            elif choice2=="{}".format(Inventory_Version_1.weapon_description_index) and Inventory_Version_1.isDemo:
                print("Description: ")      #how to make program look at descriptions for weapon...how to make it know if using spn data sets or star wars data sets...
                print(random_generation_demo_weapon_descriptions[Inventory_Version_1.weapon_description_index])
                checkpoints[last_checkpoint]()      #returns player to last checkpoint...
            elif choice2=="exit":
                checkpoints[last_checkpoint]()
            else:
                print("I don't understand that statement.  Please enter the number associated with the weapon you want to look at.")


def start():
    print("""       
        You are standing in the middle of the """ + place + """ at night.  
        There is a full moon overhead casting a faint glow on the ground in front of you.  There are trees surrounding you in every direction and span far into the night.  
        However, there seems to be traces of a path to your right.  It doesn't look to have been walked on in a long time.
        You suddenly remember why you are here.  The people of """ + place + """ have requested your help in defeating a creature that has terrorized them for many years.  Its last known location was somewhere in these woods...""")
    print("")
    flowchart.append("start")

    while True:     #Loop continuously
        choice = input()   #Get the input
        if choice in ("go on path", "go along path"):    #Correct responses...
            startpath()       #...break the loop
        elif choice == 'exit game':
            data.update({'checkpoint': 0})
            with open('save.txt', 'w') as save_file:
                json.dump(data,save_file)
            print("Saving the game...")
        elif choice == 'inventory':
            Inventory_Version_1.main1()     #Will access Inventory_Version_1 to show inventory options
            weapon_inventory_descriptions()
        else:
            print('I do not understand that statement.')    #error control

def startpath():
    print("")
    print("""
        You walk along the path, careful to not trip on any rocks or limbs along the way.  You don't get very far before seeing an object lying on the ground, shining from the moonlight filtering through the trees.  You can't make out exactly what it is, though.""")
    flowchart.append("startpath")
    print("")
    while True:
        choice = input()
        if choice in ("pick up object", "pick up the object", "look at object", "look at the object"):
            startpath_object()
        elif choice == 'exit game':
            data.update({'checkpoint': 1})
            with open('save.txt', 'w') as save_file:
                json.dump(data,save_file)
            print("Saving the game...")
        elif choice == 'inventory':
            Inventory_Version_1.main1()
            weapon_inventory_descriptions()
        else:
            print('I do not understand that statement.') 

def startpath_object():
    print("")
    print("""    
        You pick up the object and notice that it is a """ + weapon)
    print(weapon_description)
    print("")
    flowchart.append("startpath_object")

    global Inventory
    while True:
        choice = input()

        if choice in ("take " + weapon_name , "take weapon", "take the weapon"):    #allows player to specifically type in weapon name or just 'weapon'
            print("""
        You take the """ + weapon + """ and hold it tightly in your hand.""")  #ANSI codes implemented...change to yellow then back to white
            Inventory_Version_1.addToInventory(weapon_name)     #this adds the item to overall inventory list
            Inventory_Version_1.addToInventoryWeapons(weapon_name)  #adds item to inventory under 'Weapons' list
                #NOTE: this is currently hardcoded...to effectively be randomly generated, this will need to change so that there is a list of preconceived items associated with their respective types
                #e.g. if the player picked up a health item, one would need to code 'addToInventoryHealthItems'...would be better if program automatically assigned it
                #will work on this soon
            data.update({'complete inventory': Inventory_Version_1.Inventory})
            data.update({'weapons': Inventory_Version_1.InventoryWeapons})
            startpath_continue()
        elif choice in ("drop " + weapon_name, "leave " + weapon_name):
            print("     You put the " + weapon + " back on the ground.")
            startpath_continue()
        elif choice in ("go down path", "continue", "go along path", "continue down path", "continue on path"):
            startpath_continue()
        elif choice == 'exit game':
            data.update({'checkpoint': 2})
            with open('save.txt', 'w') as save_file:
                json.dump(data,save_file)
            print("Saving the game...")
        elif choice == 'inventory':
            Inventory_Version_1.main1()
            weapon_inventory_descriptions()
        else:
            print("I don't understand that statement.")

def startpath_continue():
     print("""
        You look around and see that the path still continues in front of you.  No other path is in sight and trees surround you.  The moonlight still filters through shining
        a faint light on the path ahead.""")
     print("")
     flowchart.append("startpath_continue")

     global Inventory
     while True:     
        choice = input() 
        if choice in ("go down path", "continue", "go along path", "continue down path", "continue on path"):   
            crossroads()
        elif choice == 'exit game':
            data.update({'checkpoint': 3})
            with open('save.txt', 'w') as save_file:
                json.dump(data,save_file)
            print("Saving the game...")
        elif choice == 'inventory':
            Inventory_Version_1.main1()
            weapon_inventory_descriptions()
        else:
            print('I do not understand that statement.')

def crossroads():
    print("")
    print("""
        You continue to go along the path and eventually reach the center of a crossroads.  There are 3 paths in front of you: one to the left, one to the right, and one that seems to continue from the path you are on currently.
        The middle section of the crossroads is a wide circle with a """ + trash_can +""" sitting in the center.  There is a lamp post lighting the center of the crossroads.""")
    print("")
    flowchart.append("crossroads")

    while True:
        choice = input()

        if choice in ("go to " + trash_can_name, "go to trash can"):
            print("""       You are now standing right in front of the """ + trash_can)
        elif choice in ("look inside " + trash_can_name, "look inside the " + trash_can_name):
            print("     You look inside the " + trash_can + " and see there is a " + piece_of_paper + "  at the bottom")
        elif choice in ("get " + piece_of_paper_name, "get the " + piece_of_paper_name, "pick up " + piece_of_paper_name, "pick up the " + piece_of_paper_name):
            print("""
        You pick up the """ + piece_of_paper + """ and read it.  """ + 
        """It reads: 
            Welcome to Rovivrus!  In this game, you will find there are many paths to go on.  There is no right or wrong way to play this game.  While one path might lead to something incredible, 
            another could lead to your demise.  Be cautious.  There are several others in this world, but not all will be friendly.  Be prepared for anything.
        """)
        elif choice in ("go on left path", "go down left path"):
            crossroads_left()   #demo 1 playthrough...code others in phase 2
        elif choice in ("go on right path", "go down right path"):
            crossroads_right()
        elif choice in ("go forward", "go down same path", "go on same path", "continue on same path", "continue down same path"):
            crossroads_forward()
        elif choice == 'exit game':
            data.update({'checkpoint': 4})
            with open('save.txt', 'w') as save_file:
                json.dump(data,save_file)
            print("Saving the game...")
        elif choice == 'inventory':
            Inventory_Version_1.main1()
        else:
            print("I don't understand that statement.")

def crossroads_left():
    print("")
    print("""
        You go down the path to your left, leaving the crossroads behind you.  Eventually, the light from the crossroads becomes faint.  The path in front of you is almost invisible from the pitch black darkness all around.
        Suddenly, a growling sound can be heard from in front of you, though you cannot see what is making the sound.""")
    print("")
    flowchart.append("crossroads_left")

    while True:
        choice = input()

        if choice in ("keep going forward", "go forward", "continue forward", "keep going"):
            crossroads_left1()
        elif choice in ("turn around", "turn back"):
            print("""
        You are turned around with the growling still menacingly continuing behind you.""")
        elif choice in ("go back to crossroads", "go to crossroads", "return to crossroads"):
            print("""
        You turn around and go back to the crossroads that you can barely make out in the dark from walking so far from it.  The growling continues behind you, but it eventually becomes faint.  Soon you are back at the crossroads.""")
            crossroads()
        elif choice in ("fight", "attack", "kill"):
            preattack_inventory_check()
        elif choice == 'exit game':
            data.update({'checkpoint': 5})
            with open('save.txt', 'w') as save_file:
                json.dump(data,save_file)
            print("Saving the game...")
        elif choice == 'inventory':
            Inventory_Version_1.main1()
        else:
            print("I don't understand that statement.")

def crossroads_left1():
    print("")
    print("""
        You slowly move forward towards the growling which has gotten significantly louder and more menacing now.  After a few steps, you can start to make out what seems to be a """ + enemy + """.
        A pair of glowing eyes are faint, but seem to be staring right into your soul.  The creature's growling starts to hurt your ears as it increases in volume.""")
    print("")
    flowchart.append("crossroads_left1")

    while True:
        choice = input()

        if choice in ("keep going forward", "go forward", "continue forward", "keep going"):
            crossroads_left2()
        elif choice in ("turn around", "turn back"):
            print("""
        You are turned around with the growling still menacingly continuing behind you.""")
        elif choice in ("go back to crossroads", "go to crossroads", "return to crossroads"):
            print("""
        You turn around and go back to the crossroads that you can barely make out in the dark from walking so far from it.  The growling continues behind you, but it eventually becomes faint.  Soon you are back at the crossroads.""")
            crossroads()
        elif choice in ("attack", "fight", "attack the " + enemy_name, "fight the " + enemy_name):
            attack_wolf()
        elif choice == 'exit game':
            data.update({'checkpoint': 6})
            with open('save.txt', 'w') as save_file:
                json.dump(data,save_file)
            print("Saving the game...")
        elif choice == 'inventory':
            Inventory_Version_1.main1()
        else:
            print("I don't understand that statement.")

def crossroads_left2():
    print("")
    print("""
        You continue to move forward towards the """ + enemy + """ which is still growling.  You only get three steps further before the creature suddenly lunges forward at you.  Fangs sink down into your right arm as the """
       + enemy + """ bites down hard.  The pain causes you to scream out in pain.  Blood is now all over your arm and falling to the forest floor.  The """ + enemy + """ continues to hold his grip on your arm and shows no signs of letting go.""")
    print("")
    flowchart.append("crossroads_left2")

    while True:
        choice = input()

        if choice in ("attack", "fight", "attack the " + enemy_name, "fight the " + enemy_name):
            attack_wolf()
        elif choice in ("run away", "run", "leave"):
            print("""
        You try to get away from the """ + enemy + ", but it is futile.  It continues to bite down even harder than before.  Not a few moments later, your vision starts to deteriorate and you fall into a dark abyss of nothingness...")
            death()
        elif choice == 'exit game':
            data.update({'checkpoint': 7})
            with open('save.txt', 'w') as save_file:
                json.dump(data,save_file)
            print("Saving the game...")
        elif choice == 'inventory':
            Inventory_Version_1.main1()
        else:
            print("I don't understand that statement.")

def death():
    print("")
    open('save.txt', 'w').close()   #clears all saved data from save file upon death of player
    print("You died...")
    print("Would you like to return to the main menu?  If not, the game will close.")
    print("")
    while True:
        choice = input()

        if choice in ("yes", "Y", "Yes", "y"):
            main_menu()     #allows player to restart game again
        elif choice in ("no", "N", "No", "n"):
            exit()      #closes game completely
        else:
            print("I don't understand that statement.")

def preattack_inventory_check():
    #checks if the player has any weapons available to attack with...if so, then the function continues...
    while True:
        if Inventory_Version_1.InventoryWeapons == []:
            print("You have no weapons to attack with. ")
            stack = traceback.extract_stack()
            filename, codeline, funcName, text = stack[-2]

            #print(funcName)
            checkpoints = [start, startpath, startpath_object, startpath_continue, crossroads, crossroads_left, crossroads_left1, crossroads_left2]
            #checkpoints = ["start", "startpath", "startpath_object", "startpath_continue", "crossroads", "crossroads_left", "crossroads_left1", "crossroads_left2"]
            last_checkpoint = flowchart[-1]     #gets last checkpoint in player's flowchart
            print(last_checkpoint)
            #methods = {last_checkpoint: start or startpath or crossroads_left}
            #method_name = 'start' # set by the command line options
            checkpoints[last_checkpoint]()      #automatically sends player back to checkpoint...
            if funcName in checkpoints:      #compares checkpoint name to function names
                checkpoints[last_checkpoint]()      #automatically sends player back to checkpoint...
            else:
                raise Exception("Method %s not implemented" % last_checkpoint)
        else:
            attack_wolf()

def attack_wolf():
    print("""
        What do you want to fight with?""")   #allows player to choose a weapon to fight with
    print("""
        Current weapons avaiable to use:""")   #retrieves avaiable weapons from inventory player currently has
    Inventory_Version_1.showWeapons()

    flowchart.append("attack_wolf")

    while True:
        choice = input()
        
        if choice in Inventory_Version_1.InventoryWeapons:  #read player input and match string value to inventory weapons available
            print("""
            You use the """ + choice + """ to attack the """ + enemy + """ and cause a deep gash on its right side.  Shrieking in pain and full of rage, it lunges towards you with viciousness.""")
            attack_wolf2()
        else:
            print("I don't understand that statement.")

def attack_wolf2():
    print("     Do you want to fight or flee?")

    while True:
        choice = input()

        if choice in ("fight"):
            print("""
            You attack the """ + enemy + """ again, this time using all your strength to cause a devastating blow, knocking the """ + enemy + """ to the ground.  It attempts to get up to fight again, but you finally finish it off.""")
            win()
        elif choice in ("flee"):
            print("""
            You try to run away from the """ + enemy + """ and you are almost to the end of the path where the crossroads begins when a sudden force knocks you down.  You try to get back up, but the """ + enemy + """has you pinned down.
            Pain rushes through your body as it digs its claws into you.  You are completely helpless...""")
            death()
        else:
            print("I don't understand that statement.")

def win():
    print("")
    print("""
        Congratulations!  You have fought the """ + enemy + " and lived to tell the tale.  Because of your actions, the people of " + place + " can now live in peace.")

def crossroads_left3():
    print("yay")
    
if __name__ == "__main__":
    main()

checkpoints = [start, startpath, startpath_object, startpath_continue, crossroads, crossroads_left, crossroads_left1, crossroads_left2, crossroads_left3]

def load_game():
    try:
        saved_checkpoint = int(open("save.txt").read())
        checkpoints[saved_checkpoint]()
    except FileNotFoundError:
        start()