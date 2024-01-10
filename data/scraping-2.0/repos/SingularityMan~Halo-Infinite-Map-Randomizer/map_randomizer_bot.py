"""
This code is designed to automate the process of creating a map in Halo Infinite Forge.
It is designed to be used with the Halo Infinite Forge Map Maker Bot
The bot has two modes: Manual and AI.

Manual mode allows the user to manually randomize the position of the objects.
AI mode uses GPT-4 to generate a structure in Halo Infinite Forge.

Manual mode allows the user to establish the positional parameters (x, y, z placement) of the objects.
Manual mode also allows for the user to determine exactly how many objects to add to the map.

AI mode allows GPT-4 to generate simple structures in Halo Infinite Forge and add biomes.
AI mode can also distribute the objects on the map more precisely than manual mode, i.e. creating a forest composed of clusters of trees grouped together.
"""

import pyautogui
import time
import random
import openai
import ast
import os

# OpenAI API key. Make sure it is inside a text file called 'api_key' in the same directory as this file.
# Alternatively, you may include it in your source code or environment variables.
openai.api_key = os.getenv('OPENAI_API_KEY')

if openai.api_key is None:

    with open('api_key', 'r') as f:
        openai.api_key = f.read()

    if openai.api_key == "<YOUR API KEY HERE>":
        openai.api_key = None


# Create the randomization bot
class MapMakerBot:
    def __init__(self):

        # Include ai_generate and tuple_list as a global variable
        global ai_generate
        global tuple_list

        # This is determined when the user is asked if they want to use GPT-4 to generate a structure.
        if ai_generate:
            self.ai_generate = True
        else:
            self.ai_generate = False

        # Determines how many objects to add either in manual or AI mode.
        if self.ai_generate:
            self.repetitions = len(tuple_list)
        else:
            self.repetitions = range(int(input('How many repetitions? ')))

        # Determines whether to randomize the selection of objects in manual mode, i.e. selecting different types of trees, etc.
        # You need to pay close attention to how many trees are in the object browser menu.
        self.tab_selection = input('Randomize object selection? (y/n) ')
        if self.tab_selection.lower() == 'y':
            self.tab_selection = 1
            self.tab_selection_destination_end = int(
                input('Randomize ending point for object selection (must be greater than 1)? '))
            self.tab_selection_destination = random.randint(1, self.tab_selection_destination_end)
            if self.tab_selection_destination_end < 1:
                self.tab_selection_destination_end = 1
        else:
            self.tab_selection = 0
            self.tab_selection_destination = 0

        # Determines whether to stack objects on top of each other.
        if input('Stack objects one on top of the other? (y/n) ').lower() == 'y':
            self.stack = True
        else:
            self.stack = False

        # Determines the time between each key press. (Usually between 0-1 seconds is good but you may slow it down for debugging purposes)
        self.seconds = float(input('How many milliseconds between keyboard presses (recommended between 0-1?) '))

        # Declare object properties
        self.xscale, self.yscale, self.zscale = None, None, None
        self.xscale_min, self.yscale_min, self.zscale_min, self.xscale_max, self.yscale_max, self.zscale_max = None, None, None, None, None, None
        self.pitch, self.yaw, self.roll = None, None, None
        self.pitch_min, self.yaw_min, self.roll_min, self.pitch_max, self.yaw_max, self.roll_max = None, None, None, None, None, None
        self.xpos, self.ypos, self.zpos = None, None, None
        self.xpos_min, self.ypos_min, self.zpos_min, self.xpos_max, self.ypos_max, self.zpos_max = None, None, None, None, None, None

        # Set the object properties
        self.get_object_properties()

        # Start the bot in 10 seconds.
        print('Starting in 10 seconds...')
        time.sleep(10)

    # Set object properties
    def get_object_properties(self):

        if self.ai_generate:
            object_properties = 'y'
        else:
            object_properties = input('Enable object properties? (y/n) ')

        if object_properties.lower() == 'y':
            self.object_properties = True

            # Ask whether to enable rotation modification
            rotation_enabled = input('Enable rotation modification? (y/n) ')

            # Ask for rotation values
            if rotation_enabled.lower() == 'y':

                self.pitch = input('What pitch? (you can also type "default" for default or "r" for random) ')
                self.yaw = input('What yaw? (you can also type "default" for default or "r" for random) ')
                self.roll = input('What roll? (you can also type "default" for default or "r" for random) ')

                if self.pitch != 'default' and self.pitch != 'r':
                    self.pitch = int(self.pitch)
                if self.yaw != 'default' and self.yaw != 'r':
                    self.yaw = int(self.yaw)
                if self.roll != 'default' and self.roll != 'r':
                    self.roll = int(self.roll)

            if self.ai_generate:
                position_enabled = 'y'
            else:
                position_enabled = input('Enable position modification? (y/n) ')

            # Ask for position values
            if position_enabled.lower() == 'y':

                if self.ai_generate is False:
                    self.xpos = input('Which x_position? (you can also type "default" for default or "r" for random) ')
                    self.ypos = input('Which y_position? (you can also type "default" for default or "r" for random) ')

                    if self.stack is False:
                        self.zpos = input(
                            'Which z_position? (you can also type "default" for default, "r" for random or "floor" if you want the object placed on the ground.) ')
                        if self.zpos == 'floor':
                            self.zpos = 500

                    if self.xpos != 'default' and self.xpos != 'r':
                        self.xpos = int(self.xpos)
                    elif self.xpos == 'r':
                        self.xpos_min = int(input('What is the minimum x position? '))
                        self.xpos_max = int(input('What is the maximum x position? '))

                    if self.ypos != 'default' and self.ypos != 'r':
                        self.ypos = int(self.ypos)
                    elif self.ypos == 'r':
                        self.ypos_min = int(input('What is the minimum y position? '))
                        self.ypos_max = int(input('What is the maximum y position? '))

                    if self.stack is False:
                        if self.zpos != 'default' and self.zpos != 'r':
                            self.zpos = int(self.zpos)
                        elif self.zpos == 'r':
                            self.zpos_min = int(input('What is the minimum z position? '))
                            self.zpos_max = int(input('What is the maximum z position? '))

            # Ask for scale values
            scale_enabled = input('Enable scale modification? (y/n) ')
            if scale_enabled.lower() == 'y':

                self.xscale = input('What xscale? (type "default" for default or "r" for random) ')
                self.yscale = input('What yscale? (type "default" for default or "r" for random) ')
                self.zscale = input('What zscale? (type "default" for default or "r" for random) ')

                if self.xscale != 'default' and self.xscale != 'r':
                    self.xscale = int(self.xscale)
                elif self.xscale == 'r':
                    self.xscale_min = input('What is the minimum xscale? ')
                    self.xscale_max = input('What is the maximum xscale? ')
                    self.xscale_min = int(self.xscale_min)
                    self.xscale_max = int(self.xscale_max)
                if self.yscale != 'default' and self.yscale != 'r':
                    self.yscale = int(self.yscale)
                elif self.yscale == 'r':
                    self.yscale_min = input('What is the minimum yscale? ')
                    self.yscale_max = input('What is the maximum yscale? ')
                    self.yscale_min = int(self.yscale_min)
                    self.yscale_max = int(self.yscale_max)
                if self.zscale != 'default' and self.zscale != 'r':
                    self.zscale = int(self.zscale)
                elif self.zscale == 'r':
                    self.zscale_min = input('What is the minimum zscale? ')
                    self.zscale_max = input('What is the maximum zscale? ')
                    self.zscale_min = int(self.zscale_min)
                    self.zscale_max = int(self.zscale_max)
        else:
            self.object_properties = False

    def press_key(self, key, press_count=1):
        for _ in range(press_count):
            pyautogui.keyDown(key)
            pyautogui.keyUp(key)

    def add_object(self, x_ai=None, y_ai=None, z_ai=None):

        # Open object browser menu
        self.press_key('r')

        # Randomize tab selection if enabled
        if self.tab_selection != 0:
            self.tab_selection_destination = random.randint(1, self.tab_selection_destination_end + 1)
            while self.tab_selection < self.tab_selection_destination:
                self.press_key('s')
                self.tab_selection += 1
            while self.tab_selection > self.tab_selection_destination:
                self.press_key('w')
                self.tab_selection -= 1

        self.press_key('enter')
        time.sleep(self.seconds)
        if self.object_properties or self.ai_generate:

            self.press_key('r')
            # pyautogui.mouseInfo()
            for _ in range(4):
                time.sleep(self.seconds)
                self.press_key('q')

            time.sleep(self.seconds)

            if self.xscale != 'default':
                if self.xscale is not None:
                    if self.xscale != 'r':
                        self.press_key('enter')
                        pyautogui.typewrite(str(self.xscale))
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        print('xscale', self.xscale)
                    else:
                        self.press_key('enter')
                        pyautogui.typewrite(str(random.randint(self.xscale_min, self.xscale_max)))
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        # print('xscale', self.xscale)

            self.press_key('s')
            if self.yscale != 'default':
                if self.yscale is not None:
                    if self.yscale != 'r':
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        # print('yscale', self.yscale)
                        pyautogui.typewrite(str(self.yscale))
                        self.press_key('enter')
                    else:
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        # print('yscale', self.yscale)
                        pyautogui.typewrite(str(random.randint(self.yscale_min, self.yscale_max)))
                        self.press_key('enter')

            time.sleep(self.seconds)
            self.press_key('s')
            if self.zscale != 'default':
                if self.zscale is not None:
                    if self.zscale != 'r':
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        # print('zscale', self.zscale)
                        pyautogui.typewrite(str(self.zscale))
                        self.press_key('enter')
                    else:
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        # print('zscale', self.zscale)
                        pyautogui.typewrite(str(random.randint(self.zscale_min, self.zscale_max)))
                        self.press_key('enter')

            # Input position
            for _ in range(4):
                print('pressing s')
                self.press_key('s')
                time.sleep(self.seconds)

            if x_ai is None:
                if self.xpos != 'default':
                    if self.xpos is not None:
                        if self.xpos != 'r':
                            self.press_key('enter')
                            time.sleep(self.seconds)
                            print('xpos', self.xpos)
                            pyautogui.typewrite(str(self.xpos))
                            self.press_key('enter')
                        else:
                            self.press_key('enter')
                            time.sleep(self.seconds)
                            print('xpos', self.xpos)
                            pyautogui.typewrite(str(random.randint(self.xpos_min, self.xpos_max)))
                            self.press_key('enter')
            else:
                self.press_key('enter')
                time.sleep(self.seconds)
                print('xpos', self.xpos)
                pyautogui.typewrite(str(x_ai))
                self.press_key('enter')

            time.sleep(self.seconds)
            self.press_key('s')

            if y_ai is None:
                if self.ypos != 'default':
                    if self.ypos is not None:
                        if self.ypos != 'r':
                            self.press_key('enter')
                            time.sleep(self.seconds)
                            print('ypos', self.ypos)
                            pyautogui.typewrite(str(self.ypos))
                            self.press_key('enter')
                        else:
                            self.press_key('enter')
                            time.sleep(self.seconds)
                            print('ypos', self.ypos)
                            pyautogui.typewrite(str(random.randint(self.ypos_min, self.ypos_max)))
                            self.press_key('enter')
            else:
                self.press_key('enter')
                time.sleep(self.seconds)
                print('ypos', self.ypos)
                pyautogui.typewrite(str(y_ai))
                self.press_key('enter')

            time.sleep(self.seconds)
            self.press_key('s')

            if z_ai is None:
                if self.stack is False:
                    if self.zpos != 'default':
                        if self.zpos is not None:
                            if self.zpos != 'r':
                                self.press_key('enter')
                                time.sleep(self.seconds)
                                print('zpos', self.zpos)
                                pyautogui.typewrite(str(self.zpos))
                                self.press_key('enter')
                            elif self.zpos == 'r':
                                self.press_key('enter')
                                time.sleep(self.seconds)
                                print('zpos', self.zpos)
                                pyautogui.typewrite(str(random.randint(self.zpos_min, self.zpos_max)))
                                self.press_key('enter')
                            else:
                                self.press_key('enter')
                                time.sleep(self.seconds)
                                print('zpos', self.zpos)
                                pyautogui.typewrite(str(self.zpos))
                                self.press_key('enter')
            else:
                self.press_key('enter')
                time.sleep(self.seconds)
                print('zpos', self.zpos)
                pyautogui.typewrite(str(z_ai))
                self.press_key('enter')

            time.sleep(self.seconds)
            for _ in range(2):
                self.press_key('s')
                time.sleep(self.seconds)

            if self.yaw != 'default':
                if self.yaw is not None:
                    if self.yaw != 'r':
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        print('yaw', self.yaw)
                        pyautogui.typewrite(str(self.yaw))
                        self.press_key('enter')
                    else:
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        print('yaw', self.yaw)
                        pyautogui.typewrite(str(random.randint(0, 180)))
                        self.press_key('enter')

            time.sleep(self.seconds)
            self.press_key('s')
            if self.pitch != 'default':
                if self.pitch is not None:
                    if self.pitch != 'r':
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        print('pitch', self.pitch)
                        pyautogui.typewrite(str(self.pitch))
                        self.press_key('enter')
                    else:
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        print('pitch', self.pitch)
                        pyautogui.typewrite(str(random.randint(0, 180)))
                        self.press_key('enter')

            time.sleep(self.seconds)
            self.press_key('s')
            if self.roll != 'default':
                if self.roll is not None:
                    if self.roll != 'r':
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        print('roll', self.roll)
                        pyautogui.typewrite(str(self.roll))
                        self.press_key('enter')
                    else:
                        self.press_key('enter')
                        time.sleep(self.seconds)
                        print('roll', self.roll)
                        pyautogui.typewrite(str(random.randint(0, 180)))
                        self.press_key('enter')

            time.sleep(self.seconds)
            # Press w
            for _ in range(12):
                self.press_key('w')
                time.sleep(self.seconds)
            print('pressing w')
            time.sleep(self.seconds)
            self.press_key('q')
            self.press_key('r')

        if self.stack is True:
            self.press_key('end')

    def generate_map(self, repetitions):

        for _, __ in enumerate(repetitions):

            if self.ai_generate:
                self.add_object(x_ai=__[0], y_ai=__[1], z_ai=__[2])
            else:
                self.add_object()


ai_generate = False
ai_pass = False
mode = 0
tuple_list = []

while True:

    # Use GPT-4 to create a structure in Halo Infinite Forge.
    # Skip if the user does not have an API key or declines to use GPT-4.
    if ai_generate is False:
        ai_generate = True if input(
            "Would you like to use GPT-4 to generate a structure? (y/n) ") == 'y' and openai.api_key is not None else False
        if ai_generate is False:
            print("\nNo API key included. Switching to Manual mode.\n")
        if ai_generate is True:
            mode = int(input("Which mode would you like to use? (1 = structures, 2 = environment) "))
            query = input("What would you like to build? ")

            # Architecture design (Walls, towers, etc.) it can only build simple structures.
            # If you think you can improve this prompt the be my guest. -\_(o.o)_/-
            string1 = "You are a creative AI assistant skilled in architectural design within the environment of Halo Infinite Forge. " \
                      "1. Your task is to plan and create a 3D structure by arranging a specified number of 8x8x8 crates or any other type of object with dimensions specified by the player.\n" \
                      "2. These structures could range from simple designs like walls and towers, to complex ones like castles and mazes.\n" \
                      "3. Each crate can be placed anywhere within the coordinates of -2000 to 2000 on the X, Y axis unless the user specifies otherwise, and by default on the floor of the map on the z-axis (which is usually between 500 and 490 unless the player specifies otherwise).\n" \
                      "4. You may stack objects on top of each other to form multilevel structures. The objects should be placed like lego bricks, adjacent to each other (unless otherwise specified by the user), although you may space them apart if necessary to create architectural features like rooms, halls, or corridors.\n" \
                      "5. To communicate your design, you will generate a list of comma-separated tuples containing (<x-axis integer>, <y-axis integer>, <z-axis integer>) coordinates, representing an object's position.\n" \
                      "6. Replace the letters in the tuple with an integer representing the corresponding axis.\n" \
                      "7. Replace the letters in the tuple with an integer representing the corresponding axis.\n" \
                      "8. Remember, we're working within a 3D space, so consider how your structure will look from all sides.\n" \
                      "9. Design  with intention and creativity, while keeping within the constraints of the specified number of objects and the game's coordinate system.\n" \
                      "10. Start with a foundation and build upwards, considering structural integrity, functionality, and aesthetics. Do not include any additional text or acknowledgement in your responses, only the list of coordinates.\n" \
                      "11. Make sure to contain the exact number of tuples for each object.\n\n" \
                      "Now, let's create something amazing!"

            # Environment design (Forests, glaciers, etc.)
            # This one yields better results than the architecture design prompt.
            string2 = "You are a creative AI assistant skilled in environmental design within the environment of Halo Infinite Forge.\n" \
                      "1. Your task is to plan and create a 3D structure by arranging a specified number of objects.\n" \
                      "2. These structures could range from simple designs rocks and bushes to complex ones like entire biomes, such as forests, glaciers, canyons, etc.\n" \
                      "3. Each object can be placed anywhere within the coordinates of -2000 to 2000 on the X and Y axis unless the user specifies otherwise, and whichever height the user specifies on the z-axis (although it is usually between 500 - 490).\n" \
                      "4. The distribution and number of objects will be determined by the player.\n" \
                      "5. To communicate your design, you will generate a list of (x, y, z) coordinates, each representing an object's position, with each set of coordinates enclosed in parentheses and separated by commas, replace the letters in the tuple with an integer representing the corresponding axis.\n" \
                      "6. Remember, we're working within a 3D space, so consider how your structure will look from all sides\n." \
                      "7. Design  with intention and creativity, while keeping within the constraints of the specified number of objects and the game's coordinate system.\n" \
                      "8. Do not include any additional text or acknowledgement in your responses, only the list of coordinates and separate the output in paragraphs of tuples.\n\n" \
                      "Now, let's create something amazing!"

            match mode:
                case 1:
                    mode = string1
                case 2:
                    mode = string2

            # Provides feedback in the event that GPT-4 does not generate the correct output.
            # This helps ensure GPT-4 doesn't repeat itself.
            feedback = ''
            while ai_generate and ai_pass is False:
                try:
                    ai_pass = False

                    # Tells GPT-4 to generate a structure, whether biome or structure.
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": mode},
                            {"role": "user", "content": feedback+"Build" + query}

                        ]
                    )

                    print(response.choices[0]['message']['content'])

                    response_string = response.choices[0]['message']['content']

                    # remove all whitespace
                    response_string = response_string.replace("\n", "").replace(" ", "")

                    # split the string into individual tuples
                    tuples = ast.literal_eval(response_string)

                    # create an empty list to hold the tuples
                    tuple_list = []

                    for tuple_string in tuples:
                        tuple_list.append(tuple_string)

                    ai_pass = True

                # This is usually raised when GPT-4 includes text in the output or the format is incorrect.
                except:
                    print("ERROR")
                    feedback = "The previous output returned and error: \n\n{}\n\n. Please try again and follow the user's instructions carefully.\n\n".format(tuple_list)
                    continue

    # Using the bot, breaks when player types 'n' when asked if they want to use GPT-4 instead.
    if ai_generate:
        bot = MapMakerBot()
        bot.generate_map(tuple_list)
    else:
        # Set ai_generate to False in order use the bot by manually randomizing the position of the objects.
        # Either method works well but GPT-4 allows for more controlled randomization.
        # i.e. GPT-4 ensures the structure is distributed evenly (clusters of trees evenly scattered, etc.)
        bot = MapMakerBot()
        bot.generate_map(bot.repetitions)
    if input('Would you like to add another object? (y/n) ') == 'n':
        break
    else:
        ai_generate = False
        ai_pass = False
