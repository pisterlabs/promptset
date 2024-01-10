# import openai_secret_manager
import openai
import requests
import tkinter as tk
from io import BytesIO
# from PIL import Image, ImageTk
import PIL.Image
from tkinter import *
from PIL import ImageTk, Image

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox

def gpt3(stext):
    openai.api_key = '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'
    response = openai.Completion.create(
        #        engine="davinci-instruct-beta",
        engine="text-davinci-003",
        prompt=stext,
        temperature=0.1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    content = response.choices[0].text.split('.')
    # print(content)
    return response.choices[0].text

def genSceneImage(actionevent, characters):
  queryScene = f"Write a concise prompt for the scene of {actionevent} with characters {characters} for DALL-E in less than 25 words"
  imageprompt = gpt3(queryScene)
  return imageprompt



def genImagefromScene(imageprompt):

    # Get API key
    api_key = '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'

    # # Define the story
    # story = "Once upon a time, there was a young girl named Alice who went on a journey through a magical land. She met many interesting characters, such as a rabbit in a waistcoat and a caterpillar smoking a hookah. Along the way, she faced many challenges, but she always found a way to overcome them with her courage and determination."

    # Use the OpenAI API to generate the image
    openai.api_key = api_key
    prompt = imageprompt
    # prompt = imageprompt
    response = openai.Image.create(
        prompt=prompt,
        model="image-alpha-001"
    )

    # Get the image data from the URL
    response = requests.get(response["data"][0]["url"])
    img_data = PIL.Image.open(BytesIO(response.content))

    return img_data
# Get API key
# api_key = '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'

# Define the story
# story = "Once upon a time, there was a young girl named Alice who went on a journey through a magical land. She met many interesting characters, such as a rabbit in a waistcoat and a caterpillar smoking a hookah. Along the way, she faced many challenges, but she always found a way to overcome them with her courage and determination."

# Use the OpenAI API to generate the image
# openai.api_key = api_key
# prompt = (f"generate a image with Dall-E based on the following story: {story}")

# prompt = genSceneImage("The rabbit ran up the hill", characters=["rabbit", "coyote"])
# print(prompt)



# import tkinter as tk
# from tkinter import *
# from tkinter import ttk
from tkinter import messagebox

# from PIL import ImageTk, Image

personality_list = ["bpsc", "bpcs", "cbsp", "cpsb", "sbcp", "sbpc", "pcbs", "pcsb", "scbp", "scpb", "cbps", "csbp",
                    "bpsc", "bpsc", "spbc", "spcb"]
story_image = True  # Just a placeholder, we need to build the image using PIL


class App:
    def __init__(self, story, characters, story_image):
        # Create the main window
        self.window = tk.Tk()
        self.window.title("SIBILANCE AI - STORY GENERATOR")
        # Changing the background color of the window
        self.window.configure(background='ghost white')

        self.lb1 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, text=" CHARACTER ", bg="CadetBlue1", font=("Georgia, 14"))
        self.lb1.place(x=530, y=20)

        self.lb2 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, text="   PERSONALITY     ", bg="CadetBlue1", font=("Georgia, 14"))
        self.lb2.place(x=670, y=20)

        self.tb1 = tk.Text(self.window, bd=5)
        self.tb1.place(x=530, y=60, height=30, width=110)

        self.cb1 = ttk.Combobox(self.window, width=5, font=("Georgia, 12"))
        self.cb1['values'] = (personality_list)
        self.cb1.place(x=670, y=60, width=150)

        self.tb2 = tk.Text(self.window, bd=5)
        self.tb2.place(x=530, y=100, height=30, width=110)

        self.cb2 = ttk.Combobox(self.window, width=5, font=("Georgia, 12"))
        self.cb2['values'] = (personality_list)
        self.cb2.place(x=670, y=100, width=150)

        self.tb3 = tk.Text(self.window, bd=5)
        self.tb3.place(x=530, y=140, height=30, width=110)

        self.cb3 = ttk.Combobox(self.window, width=5, font=("Georgia, 12"))
        self.cb3['values'] = (personality_list)
        self.cb3.place(x=670, y=140, width=150)

        self.lb6 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, text="   STORY   ", bg="CadetBlue1", font=("Georgia, 14"))
        self.lb6.place(x=20, y=240)

        self.tb4 = tk.Text(self.window, bd=5)
        self.tb4.place(x=20, y=280, height=150, width=500)

        self.lb9 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, text="   ACTUAL IMAGE   ",
                            font=("Helvetica", 12), bg='#F0F0F0')
        self.lb9.place(x=20, y=20, height=200, width=500)

        self.lb10 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE,
                             text="   OPTIONALLY SUGGEST NEXT ACTION EVENT \n   WRITE A FULL MEANINGFUL SENTENCE   ",
                             bg="CadetBlue1", font=("Georgia, 13"))
        self.lb10.place(x=20, y=450)

        self.tb5 = tk.Text(self.window, bd=5)
        self.tb5.place(x=20, y=500, height=30, width=500)

        # self.btn1 = tk.Button(self.window, text="   DETERMINE NEXT ACTION EVENT   ", fg="black", font=("Georgia, 13"),
        #                       bg="gold", relief=tk.RAISED, command=self.getUIDataAndChooseAction)
        self.btn1 = tk.Button(self.window, text="   DETERMINE NEXT ACTION EVENT   ", fg="black", font=("Georgia, 13"),
                              bg="gold", command=self.getUIDataAndChooseAction)
        self.btn1.place(x=50, y=540)

        self.window.geometry("1000x600+10+10")
        self.window.mainloop()

    ##############################

    def getUIDataAndChooseAction(self):
        global first_time

        st_character1 = self.tb1.get("1.0", "end-1c")
        self.lb3 = tk.Label(self.window, bd=5, fg="#000", font=("Georgia, 12"), relief=tk.GROOVE, text=st_character1)
        self.lb3.place(x=530, y=60, height=30, width=110)

        st_character2 = self.tb2.get("1.0", "end-1c")
        self.lb4 = tk.Label(self.window, bd=5, fg="#000", font=("Georgia, 12"), relief=tk.GROOVE, text=st_character2)
        self.lb4.place(x=530, y=100, height=30, width=110)

        st_character3 = self.tb3.get("1.0", "end-1c")
        self.lb5 = tk.Label(self.window, bd=5, fg="#000", font=("Georgia, 12"), relief=tk.GROOVE, text=st_character3)
        self.lb5.place(x=530, y=140, height=30, width=110)

        characters = [st_character1, st_character2, st_character3]
        print(characters)

        st_personality_character1 = self.cb1.get()
        print(st_personality_character1)

        st_personality_character2 = self.cb2.get()
        print(st_personality_character2)

        st_personality_character3 = self.cb3.get()
        print(st_personality_character3)

        personality_characters = [st_personality_character1, st_personality_character2, st_personality_character3]
        print(personality_characters)

        # for character in characters:

        character_personality_tuple_list = self.createTuple(characters, personality_characters)
        print(character_personality_tuple_list)

        # global story
        # if first_time:
        #     story = self.tb4.get("1.0", "end-1c")
        #     print(story)
            # story = story = story.replace("\n", " ")
            # self.lb7 = tk.Label(self.window, bd=5, fg="#f0f", relief=tk.GROOVE, text=story)
            # self.lb7.place(x=20, y=280, height=150, width=500)

        user_suggested_next_action_event = self.tb5.get("1.0", "end-1c")
        user_suggestion = False
        if user_suggested_next_action_event != "":
            user_suggestion = True
        print(user_suggestion)
        print(user_suggested_next_action_event)

        # Use the following information coming from the UI and call appropriate functions
        # characters - this is a list
        # personality_characters - this is a tuple (char:personality) - The user could have changed the personality different from previous screen
        # story - free text with /n chars removed. Suggest whether we need to remove tab chars /t
        # user_suggestion - this is a flag - True or False
        # user_suggested_next_action_event - free text - this has actual suggested action event by the user_suggestion

        # Use the values gotten from your function modules to set the following values to display on the UI
        # characters = characters
        # story = story

        # story_image - this is a png image file for the story - very first screen, this will be blank - uncomment following two lines after setting up story_image value
        # self.lb9 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, image=story_image, font=("Helvetica", 12), bg='#F0F0F0')
        # self.lb9.place(x=320, y=20, height=200, width=450)
        # Add a Scrollbar(vertical)

        # Trying scrollable story - IT-3
        # v = Scrollbar(self.window, orient='vertical')
        # v.pack(side=RIGHT, fill='y')
        # v.destroy()

        # if first_time:
        #     initTransitionProbs(characters=characters)
        #     global current_state
        #     # initialize current state randomly for now
        #     for character in characters:
        #         current_state[character] = random.choice(['BP', 'BS', 'CP', 'CS'])

        #     # initalize behavior state for all characters
        #     global behavior_states
        #     for character in characters:
        #         behavior_states[character] = ['BP', 'BS', 'CP', 'CS']

            # Trying scrollable story - IT-3
            # v = Scrollbar(self.window, orient='vertical')
            # v.pack(side=RIGHT, fill='y')
        first_time = False

        # assign the characters to the personalities picked
        # for character, personality in character_personality_tuple_list:
        #     assignPersonalitytoCharacter(character, personality)

        # next_action, actionindex, eventlist = determine_next_action(story=story, characters=characters,
        #                                                             current_state=current_state,
        #                                                             behavior_states=behavior_states)
        # nextevent = update_story_data(next_action, actionindex=actionindex, eventlist=eventlist, story=story,
        #                               characters=characters)

        # # update the story with next event
        # story = story + nextevent
        # print("story ", story)
        # self.lb7 = tk.Label(self.window, bd=5, fg="#f0f", padx=5, pady=5, relief=tk.GROOVE, text=story)
        # self.lb7.place(x=20, y=280, height=150, width=500)
        # win1=App(story, characters, story_image)

        # Trying scrollable story - IT-3
        # v = Scrollbar(self.window, orient='vertical')
        # v.pack(side=RIGHT, fill='y')

        # Add a text widget
        # text = Text(self.window, font=("Georgia, 12"), yscrollcommand=v.set)
        text = Text(self.window, font=("Georgia, 12"), padx=5, pady=5, selectborderwidth=50, wrap=tk.WORD)
        # text.insert(END, story)

        # Attach the scrollbar with the text widget
        # v.config(command=text.yview)
        text.place(x=20, y=280, height=150, width=500)

        # Trying scrollable story - IT-3 - End

        # Get the image for nextevent and display in a label lb9
        # Start

        # try: 
        #     nexteventimg = eventlist[actionindex]
        # except:
        #     nexteventimg = eventlist[len(eventlist) - 1]

        # print("next event for img", nexteventimg)
        # imageprompt = genSceneImage(nexteventimg, characters)
        # print("imageprompt", imageprompt)
        # Split imageprompt into sentences
        # # sentences = re.split(r' *[\.\?!][\'"\)\]]* *', imageprompt)
        # print("imageprint: " + imageprompt)
        # print("sentences[0]")
        # # print(sentences[0])
        #
        img_data = genImagefromScene(imageprompt="dog walking on a beach")
        print("img_data ", img_data)
        # Add the image to the label
        # img = ImageTk.PhotoImage(img_data)

        # image = Image.open(img)
        img_data = img_data.resize((500, 200))

        # photo = ImageTk.PhotoImage(image)
        self.img = ImageTk.PhotoImage(img_data)
        
        # self.lb9 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, image=img, font=("Helvetica", 12), bg='#F0F0F0')
        self.lb9 = tk.Label(self.window, image=self.img)
        self.lb9.place(x=20, y=20, height=200, width=500)
        # self.lb9.pack()

        # End

        return

    # Utility Function - To create the tuples using two lists
    def createTuple(self, list1, list2):

        tuple_list = []
        for i in range(max((len(list1), len(list2)))):

            while True:
                try:
                    tup = (list1[i], list2[i])
                except IndexError:
                    if len(list1) > len(list2):
                        list2.append('')
                        tup = (list1[i], list2[i])
                    elif len(list1) < len(list2):
                        list1.append('')
                        tup = (list1[i], list2[i])
                    continue

                tuple_list.append(tup)
                break
        return tuple_list

    # def findNextActionEvent(self):
    #     st_character1 = self.tb1.get("1.0", "end-1c")
    #     self.lb3 = tk.Label(self.window, bd=5, fg="#000", relief=tk.GROOVE, text=st_character1)
    #     self.lb3.place(x=20, y=60, height=30, width=110)
    #
    #     st_character2 = self.tb2.get("1.0", "end-1c")
    #     self.lb4 = tk.Label(self.window, bd=5, fg="#000", relief=tk.GROOVE, text=st_character2)
    #     self.lb4.place(x=20, y=100, height=30, width=110)
    #
    #     st_character3 = self.tb3.get("1.0", "end-1c")
    #     self.lb5 = tk.Label(self.window, bd=5, fg="#000", relief=tk.GROOVE, text=st_character3)
    #     self.lb5.place(x=20, y=140, height=30, width=110)
    #
    #     characters = [st_character1, st_character2, st_character3]
    #     print(characters)
    #
    #     # BUILD AN ARRAY OF TUPLES OF CHARACTER AND PERSONALITY
    #
    #     # CAPTURE USER ENTERED STORY IN THE VERY FIRST TIME AND THEN JUST SEND THE STORY TO AJJU FN MODULE
    #     story = self.tb4.get("1.0", "end-1c")
    #     print(story)
    #
    #     #     sentence = self.txt1.get(1.0, "end-1c")
    #     #     word_list = sentence.split()
    #     #     story = self.txt21.get(1.0, "end")
    #     #     story1=story
    #     #     #story = story.replace("\n", " ")
    #     #     characters = word_list
    #     #
    #     #     self.lbl4=tk.Label(self.window, bd=5, relief=tk.RIDGE, justify=tk.LEFT, text=story)
    #     #     self.lbl4.place(x=140, y=60, height=150, width=500)
    #     #     self.lbl4["text"] = story1
    #     #     self.cb1["values"] = characters
    #     #
    #     #     print(sentence)
    #     #     print(word_list)
    #     #     print(word_list[1])
    #     #     print(story)
    #     return


import random


def main():
    # TODO

    current_state = {}
    # current_state = {'Alice': 'BP','Bob': 'CS'}
    behavior_states = {}
    # behavior_states = {'Alice': ['BP', 'BS', 'CP', 'CS'],'Bob': ['BP', 'BS', 'CP', 'CS']}
    # story = "Alice and Bob are going to a grocery store."
    characters = []
    story = ""
    story_image = ""  # supposed to be a png file from DallE
    setting = ""
    genre = ""
    worldbuilding = {"setting": setting, "genre": genre}

    win1 = App(story, characters, story_image)  # UI
    # if first_time:
    #   # initialize current state randomly for now
    #   for character in characters:
    #     current_state[character] = random.choice(['BP', 'BS', 'CP', 'CS'])

    #   # initalize behavior state for all characters
    #   for character in characters:
    #     behavior_states[character] = ['BP', 'BS', 'CP', 'CS']
    # first_time = False

    # next_action, actionindex, eventlist = determine_next_action(story="Alice and Bob are going to a grocery store.", characters=["Alice", "Bob"], current_state=current_state, behavior_states=behavior_states)
    # nextevent = update_story_data(next_action, actionindex=actionindex, eventlist=eventlist, story=story, characters=characters)

    # # update the story with next event
    # story = story + nextevent
    # win1=App(story, characters, story_image)

    # print("eventlist", eventlist)
    # print("next_action", next_action)
    # print("action_index", actionindex)
    # print("nextevent", nextevent)


if __name__ == "__main__":
    main()





# response = openai.Image.create(
#     prompt=prompt,
#     model="image-alpha-001"
# )

# Get the image data from the URL
# response = requests.get(response["data"][0]["url"])
# img_data = Image.open(BytesIO(response.content))

# img_data = genImagefromScene(imageprompt=prompt)

# # Create a Tkinter window
# root = tk.Tk()
# root.title("Dall-E Image")

# # Create a label to display the image
# img = ImageTk.PhotoImage(img_data)
# img_label = tk.Label(root, image=img)
# img_label.pack()

# root.mainloop()