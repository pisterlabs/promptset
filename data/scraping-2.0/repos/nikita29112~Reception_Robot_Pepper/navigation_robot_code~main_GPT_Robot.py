"""
This is the main file for the GPT Robot application.
"""
import json
import csv
import random
import time
from datetime import datetime
import logging

import openai
import tiktoken
import tkinter as tk
from tkinter import *
from tkinter import ttk
from threading import Thread
import signal

import numpy as np
import pyaudio

from rating_screen import Rating
from sic_framework.core.application import SICApplication
from sic_framework.devices.common_naoqi.naoqi_speakers import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.pepper_tablet import NaoqiTabletService, NaoqiLoadUrl
from sic_framework.devices.pepper import Pepper
from sic_framework.services.dialogflow.dialogflow_service import DialogflowService, DialogflowConf, GetIntentRequest, \
    RecognitionResult, QueryResult

from GPTChat_Prompt import GPTPrompt, get_completion_and_token_count
from html_pics import Maps
from map_img_urls import MapUrls

from sic_framework.services.webserver.webserver_service import WebserverConf, WebserverService, GetWebText
from sic_framework.devices.common_naoqi.nao_motion import NaoPostureRequest, NaoRestRequest, NaoWakeUpRequest, \
    AnimationRequest

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") # Used to keep track of the number of tokens used
max_tokens = 110

# Flag to indicate if the program is stopped
stop_flag = False


# Create a signal handler to handle the interrupt signal and save data
def handle_interrupt(signal, frame):
    global stop_flag
    print("Program interrupted. Saving data...")
    stop_flag = True


# Register the interrupt signal handler
signal.signal(signal.SIGINT, handle_interrupt)


# function read json file as dictionary
def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


# function to get stop words from file, Note: Not used, Dialogflow intent is used instead
def create_stoplist(file):
    my_file = open(file, "r")
    content = my_file.read()
    stop_list = content.split(",")
    my_file.close()
    return stop_list


# function to save interaction to csv file
# NOTE: Change filename after each run
def save_interaction_to_csv(row):
    with open('Interaction_files/interaction_05_07_test.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


# Saves the interaction list to a file
def save_list_to_file(interaction_list, filename):
    with open(filename, 'w') as f:
        for item in interaction_list:
            f.write("%s\n" % item)


class GPTPepper(SICApplication):

    def __init__(self):
        super().__init__()
        self.text_input_entry = None
        self.pepper_tablet_connector = None  # variable to connect to Pepper's tablet
        self.webserver_connector = None  # variable to connect to webserver
        self.pepper = Pepper(device_id='pepper', application=self)  # create Pepper object
        keyfile_json = json.load(open('dialogflow_keyfile.json'))  # load Dialogflow key json file
        conf = DialogflowConf(keyfile_json=keyfile_json, project_id='dialogflow_project_id', sample_rate_hertz=16000)  # connect to Dialogflow
        mic_service = self.pepper.mic  # connect to Pepper's microphone
        self.dialogflow = self.connect(DialogflowService, device_id='local', inputs_to_service=[mic_service], log_level=logging.INFO, conf=conf)  # connect to Dialogflow service
        self.user_response = self.dialogflow.register_callback(self.on_dialog) #None  # variable to save user input
        self.interaction = []  # Ongoing list tracking interaction between user and robot
        openai.api_key = "sk-OPENAI-KEY"
        self.user_model = {}  # dictionary saving user input # NOTE: Not used
        self.robot_model = {}  # dictionary saving robot output # NOTE: Not used
        self.places = read_json("places_directions.json")  # dictionary with places and directions # NOTE: Change to places_direction_mainstreet.json if Robot is placed at Main Street entrance
        self.conversation = GPTPrompt()  # create GPTPrompt object
        self.stop_list = create_stoplist("stoplist.txt")  # create list of stop words/phrases # NOTE: Not used
        self.number_of_people_options = ['1', '2', '3', '4', '5']  # Drop down list to record number of people, used in log_window()
        web_conf = WebserverConf(host="0.0.0.0", port=8080) # connecting localhost to webserver service
        self.webserver_connector = self.connect(WebserverService, device_id='web', inputs_to_service=[self], log_level=logging.INFO, conf=web_conf)
        self.pepper_tablet_connector = self.connect(NaoqiTabletService, device_id='pepper', inputs_to_service=[self]) # connecting Pepper's tablet with tablet service
        self.smiley1, self.smiley2, self.smiley3, self.smiley4, self.smiley5, self.smiley6, self.smiley7 = Rating.return_rating()  # create rating smileys to display on Rating screen # NOTE: Not used

    # function to get transcript from Dialogflow
    def on_dialog(self, message):
        if message.response:
            # print(message.response.recognition_result.transcript)
            if message.response.recognition_result.is_final:
                self.user_response = message.response.recognition_result.transcript
                print('Transcript: ', self.user_response)

                return self.user_response

    # function to end conversation
    def close_conversation(self):
        # NOTE: Uncomment this to display Rating screen on tablet at the end of conversation
        # self.rating_screen()
        self.interaction.append("END")
        self.save_to_csv('', 'END', '', '', '', '', '')

    def display_standby_url(self):
        # web_conf = WebserverConf(host="0.0.0.0", port=8080)
        # self.webserver_connector = self.connect(WebserverService, device_id='web', inputs_to_service=[self],
        #                                         log_level=logging.INFO, conf=web_conf)
        # self.pepper_tablet_connector = self.connect(NaoqiTabletService, device_id='pepper', inputs_to_service=[self])
        # web_url = "https://prnt.sc/gIqULw_mYA8v"
        web_url = "https://prnt.sc/lk-HffAQfy8y"
        self.pepper_tablet_connector.send_message(NaoqiLoadUrl(web_url))

    def display_html_on_tablet(self, html_file):
        # display html on tablet
        # web_conf = WebserverConf(host="0.0.0.0", port=8080)
        # self.webserver_connector = self.connect(WebserverService, device_id='web', inputs_to_service=[self],
        #                                         log_level=logging.INFO, conf=web_conf)
        # self.pepper_tablet_connector = self.connect(NaoqiTabletService, device_id='pepper', inputs_to_service=[self])
        # web_url = "http://192.168.0.208:8080/"  # NOTE: Change to IP of Laptop in use
        web_url = "http://10.15.0.120:8080/"
        # send html to WebserverService
        # with open(html_file) as file:
        #     file_data = file.read()
        self.webserver_connector.send_message(GetWebText(html_file))

        # send url to NaoqiTabletService in order to display it on a pepper's tablet
        self.pepper_tablet_connector.send_message(NaoqiLoadUrl(web_url))
        # time.sleep(3)

    def display_url_on_tablet(self, url):
        # display url on tablet
        web_url = url
        self.pepper_tablet_connector.send_message(NaoqiLoadUrl(web_url))
        # time.sleep(3)

    # HTML script to display Rating screen
    def rating_screen(self):
        # web_conf = WebserverConf(host="0.0.0.0", port=8080)
        # self.webserver_connector = self.connect(WebserverService, device_id='web', inputs_to_service=[self],
        #                                         log_level=logging.INFO, conf=web_conf)
        # self.pepper_tablet_connector = self.connect(NaoqiTabletService, device_id='pepper', inputs_to_service=[self])
        # web_url = "http://192.168.0.208:8080/"  # NOTE: Change to IP of Laptop in use
        web_url = "http://10.15.0.120:8080/"
        rating_html = '<nav class="navbar mb-5">' \
                      '<div class="navbar-brand listening_icon"></div>' \
                      '<div class="navbar-nav vu_logo"></div>' \
                      '</nav>' + '<main class="container text-center">' '<h1>' + 'Please give me a rating by clicking one of ' \
                                                                                 'the smileys on my tablet' + '</h1>' + \
                      self.smiley1 + self.smiley2 + self.smiley3 + self.smiley4 + self.smiley5 + self.smiley6 + self.smiley7 \
                      + '</main>' + '<footer class="fixed-bottom">' \
                                    '<p class="lead bg-light text-center speech_text"></p>' \
                                    '</footer>'
        self.pepper.text_to_speech.request(NaoqiTextToSpeechRequest('Please give me a rating by clicking one of the '
                                                                    'smileys on my tablet!'))
        self.webserver_connector.send_message(GetWebText(rating_html))
        self.pepper_tablet_connector.send_message(NaoqiLoadUrl(web_url))
        # time.sleep(3)

    # function to save interaction to csv according to columns 'Date-Time', 'User Input', 'Robot Response',
    # 'Button Clicked', 'Number of People', 'comment', and 'Number of Conversation turns':
    def save_to_csv(self, timestamp, user_input, robot_response, button_value, num_people, num_turns, comment):
        # self.interaction.append([timestamp, user_input, robot_response, button_value, num_people, num_turns, comment])
        save_interaction_to_csv([timestamp, user_input, robot_response, button_value, num_people, num_turns, comment])

    # functions for button click events on log window

    # Implements invite motion (Waving) and plays invite message
    def invite_clicked(self):
        # self.pepper.motion.request(NaoPostureRequest('Stand', 0.5))
        # Different invite messages are randomly chosen:
        # NOTE: These Invite messages are not used, since they are too long
        # invite_list = ['Hello! Welcome to the New building! If you have any questions about the campus or need '
        #                'assistance, feel free to chat with me',
        #                "Hi there! Looking for some information or directions? I'm Pepper, your trusty guide. Feel "
        #                "free to ask me anything!",
        #                "Hi! I'm Pepper, your friendly robot companion. If you have any queries or need directions, "
        #                "just let me know. I'm here to chat!",
        #                "Greetings! Need some guidance around the campus? I'm Pepper, ready to assist you. Just ask me."]

        invite_list = ["Hi! Need directions or just feel like chatting? I'm here for you!",
                       "Greetings! Looking for some information or directions? Just ask me.",
                       # "Hey, feel like having a conversation? I'm here to listen and chat!",
                       "Hi! I'm Pepper, your friendly robot companion. I'm here to help! "]

        invite_text = random.choice(invite_list)
        self.display_standby_url() # Displays welcome message on tablet
        self.pepper.text_to_speech.request(NaoqiTextToSpeechRequest(invite_text))
        messages = self.conversation.initialize_conversation() # Initializes GPT conversation
        self.conversation.add_to_conv_pepper(messages, invite_text)
        self.conversation.initialize_conversation()
        self.pepper.motion.request(AnimationRequest('animations/Stand/Gestures/Hey_1'))
        print("motion done")
        self.interaction.append('INVITE: ' + invite_text)
        self.save_to_csv('', '', '', 'INVITE: ' + invite_text, '', '', '')
        print('Invite clicked')

    # Button to track when a New Interaction is started
    def newperson_clicked(self):
        number_of_people = self.number_of_people_var.get()
        comment = self.text_input_var.get()
        self.interaction.append('NEW PERSON')
        self.save_to_csv('', '', '', 'NEW PERSON', number_of_people, '', comment)
        self.number_of_people_var.set(self.number_of_people_options[0])
        self.text_input_var.set('')
        self.text_input_entry.delete(0, tk.END)  # Clear the text input field
        print("NEW PERSON")

    # Button to track when a person Ignores Pepper
    def ignore_clicked(self):
        comment = self.text_input_var.get()
        self.interaction.append('IGNORED')
        self.save_to_csv('', '', '', 'IGNORED', '', '', comment)
        self.text_input_entry.delete(0, tk.END)  # Clear the text input field
        print("IGNORED")

    # Button to track when a person is Leaves mid-conversation
    def dropout_clicked(self):
        comment = self.text_input_var.get()
        self.interaction.append('DROPOUT')
        self.save_to_csv('', '', '', 'DROPOUT', '', '', comment)
        self.text_input_var.set('')
        self.text_input_entry.delete(0, tk.END)  # Clear the text input field
        print("DROPOUT")

    def quit_clicked(self):
        comment = self.text_input_var.get()
        self.interaction.append('QUIT')
        self.save_to_csv('', '', '', 'QUIT', '', '', comment)
        self.text_input_var.set('')
        self.text_input_entry.delete(0, tk.END)  # Clear the text input field
        print('QUIT')
        GPTPepper().stop()  # Stops the application

    # Supporting functions to save values of Number of People and Comment
    def save_number_of_people(self, value):
        comment = self.text_input_var.get()
        self.interaction.append(f'Number of People: {value}')
        self.number_of_people_var.set(value)
        self.save_to_csv('', '', '', '', value, '', comment)  # Save the selected value to CSV
        self.text_input_var.set('')
        self.text_input_entry.delete(0, tk.END)  # Clear the text input field

    def save_comment(self):
        comment = self.text_input_var.get()
        self.interaction.append(f'Comments: {comment}')
        self.save_to_csv('', '', '', '', '', '', comment)
        self.text_input_var.set('')
        self.text_input_entry.delete(0, tk.END)  # Clear the text input field
        print('Comment: ', comment)

    # Button to clear the Comment field
    def clear_clicked(self):
        comment = self.text_input_var.get()
        self.interaction.append(f'Comments: {comment}')
        self.save_to_csv('', '', '', '', '', '', comment)
        self.text_input_var.set('')
        self.text_input_entry.delete(0, tk.END)  # Clear the text input field
        print('Comment: ', comment)

    # Button to track when a person Acknowledges Pepper (Takes photo/video, waves, uses tablet, etc.) but does not speak to Pepper
    def acknowledge_clicked(self):
        comment = self.text_input_var.get()
        self.interaction.append('ACKNOWLEDGED')
        self.save_to_csv('', '', '', 'ACKNOWLEDGED', '', '', comment)
        self.text_input_entry.delete(0, tk.END)  # Clear the text input field
        print("ACKNOWLEDGED")

    # Log window to record Ignore Rate, Dropout Rate, Number of People and Comments during experiment
    def log_window(self):
        # window for Wizard-of-Oz logging procedure
        window = Tk()
        style = ttk.Style()
        # window size
        window.geometry("700x500")

        # window configure
        window.configure(bg="white")

        # Create buttons and labels
        head_label = tk.Label(window, text="Logging Screen", fg="black", font=("Calibri", 16, "bold"))
        head_label.pack()

        style.configure('NewPerson.TButton', border=8, borderwidth=4, relief=tk.RAISED, font=('Calibri', 12),
                        foreground='black', background='green')
        style.configure('Ignore.TButton', border=8, borderwidth=4, relief=tk.RAISED, font=('Calibri', 12),
                        foreground='black', background='yellow')
        style.configure('Dropout.TButton', border=8, borderwidth=4, relief=tk.RAISED, font=('Calibri', 12),
                        foreground='black', background='red')
        style.configure('QUIT.TButton', border=8, borderwidth=4, relief=tk.RAISED, font=('Calibri', 12),
                        foreground='black', background='blue')

        invite_button = ttk.Button(window, text="Invite People", command=self.invite_clicked,
                                   style='NewPerson.TButton')
        invite_button.place(x=300, y=50)

        new_person_button = ttk.Button(window, text="New Person", command=self.newperson_clicked,
                                       style='NewPerson.TButton')
        new_person_button.place(x=100, y=100)

        number_of_people_label = tk.Label(window, text="Number of People:", fg="black", font=("Calibri", 12))
        number_of_people_label.place(x=400, y=100)

        self.number_of_people_var = tk.StringVar(window)
        number_of_people_dropdown = OptionMenu(window, self.number_of_people_var, *self.number_of_people_options)
        number_of_people_dropdown.config(width=5)
        number_of_people_dropdown.place(x=550, y=100)

        ignore_button = ttk.Button(window, text="Ignore", command=self.ignore_clicked, style='Ignore.TButton')
        ignore_button.place(x=100, y=200)

        acknowledge_button = ttk.Button(window, text="Acknowledge", command=self.acknowledge_clicked, style='Ignore.TButton')
        acknowledge_button.place(x=200, y=200)

        dropout_button = ttk.Button(window, text="Drop out", command=self.dropout_clicked, style='Dropout.TButton')
        dropout_button.place(x=400, y=200)

        text_input_label = tk.Label(window, text="Comments:", fg="black", font=("Calibri", 12))
        text_input_label.place(x=200, y=300)

        self.text_input_var = tk.StringVar(window)
        self.text_input_entry = ttk.Entry(window, textvariable=self.text_input_var)
        self.text_input_entry.place(x=300, y=300)

        clear_text_button = ttk.Button(window, text="CLEAR TEXT", command=self.clear_clicked, style='QUIT.TButton')
        clear_text_button.place(x=300, y=400)

        quit_button = ttk.Button(window, text="QUIT", command=self.quit_clicked, style='QUIT.TButton')
        quit_button.place(x=320, y=450)

        window.mainloop()

    # Starts the interface window in a separate thread
    def start_interface(self):
        t = Thread(target=self.log_window)
        t.daemon = True  # Sets thread as a daemon to exit when the main program ends
        t.start()
        # self.log_window()

    # run function
    def run(self) -> None:
        self.pepper.motion.request(NaoPostureRequest('Stand', 0.5))  # plays stand up motion when the application starts
        # self.user_response = None  # reset user response
        # pepper = Pepper(device_id='pepper', application=self)  # connect to Pepper
        # keyfile_json = json.load(open('navigation-robot-pepper-384110-14bd79fc02e9.json'))  # load Dialogflow key json file
        # conf = DialogflowConf(keyfile_json=keyfile_json, project_id='navigation-robot-pepper-384110', sample_rate_hertz=16000)  # connect to Dialogflow
        # mic_service = self.pepper.mic  # connect to Pepper's microphone
        # dialogflow = self.connect(DialogflowService, device_id='local', inputs_to_service=[mic_service],log_level=logging.INFO, conf=conf)  # connect to Dialogflow service
        # self.user_response = dialogflow.register_callback(self.on_dialog)  # register callback function, stores transcript of user input
        messages = self.conversation.initialize_conversation()  # initialize conversation
        self.interaction.append("START INTERACTION")
        print(" -- START CONVERSATION -- ")  # start conversation

        # Start the log interface window in a separate thread
        self.start_interface()

        x = np.random.randint(10000)

        for i in range(1000):
            print(" ----- Conversation turn", i)  # print conversation turn
            # self.display_standby_url()  # display standby image (Welcome message) on tablet
            self.user_response = self.dialogflow.register_callback(self.on_dialog)  # get user input from Dialogflow
            dialogflow_reply = self.dialogflow.request(GetIntentRequest(x))  # get Dialogflow reply, intent and entities

            if dialogflow_reply.response.query_result.intent:
                intent_name = dialogflow_reply.response.query_result.intent.display_name
                print(f"INTENT '{intent_name}'")
                if intent_name == "answer_greeting":
                    print('GREETING INTENT')
                    self.interaction.append('Greeting Intent')

                elif intent_name == "answer_place":  # if intent is answer_place, display map of place
                    place_reg = dialogflow_reply.response.query_result.parameters["places"]
                    print('Place recognized: ', place_reg)
                    self.interaction.append('Place Recognized: ' + place_reg)
                    # NOTE: Load HTML or URL map file to tablet (depending on connection)
                    # map_html = Maps.show_maps(place_reg)
                    map_url = MapUrls.display_map_url(place_reg)
                    #  Load HTML map file to tablet
                    # self.display_html_on_tablet(map_html)
                    # OR display map URL on tablet (depending on connection)
                    self.display_url_on_tablet(map_url)

                elif intent_name == 'stop_conversation':
                    self.close_conversation()  # end conversation and ask for rating
                    print('END CONVERSATION INTENT')
                    self.interaction.append('Conversation Complete: User ended interaction')

            # messages = self.conversation.initialize_conversation()  # initialize conversation

            # self.user_response = self.dialogflow.register_callback(self.on_dialog)

            self.conversation.add_to_conv(messages, self.user_response)  # add user input to conversation

            pepper_response, token_count = get_completion_and_token_count(messages)  # get robot response from GPT-3

            self.pepper.text_to_speech.request(NaoqiTextToSpeechRequest(pepper_response))  # robot speaks

            self.conversation.add_to_conv_pepper(messages, pepper_response)  # add robot response to messages

            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")  # get timestamp

            self.save_to_csv(timestamp, self.user_response, pepper_response, '', '', i, '')  # save to csv file

            # if total tokens used is greater than 3000, removes the first message from the 'message' list
            if token_count > 3000:
                tokens_removed = 0
                while tokens_removed < max_tokens:
                    # the index of message deleted, index 0 - system message
                    tokens_removed += len(encoding.encode(messages[4]["content"]))
                    del messages[4]

            print("User: ", self.user_response)
            print("Pepper: ", pepper_response)

            # self.interaction.append(self.conversation.get_conv())
            self.interaction.append(f'USER: {self.user_response}')
            self.interaction.append(f'Pepper: {pepper_response}')
            self.interaction.append('')
            # save_interaction_to_csv(row=self.interaction)  # save interaction to csv file
            save_list_to_file(self.interaction, 'Interaction_files/interaction_text_05_07_test.txt')

            # for s in range(len(self.stop_list)):  # check if user input contains stop words
            #     if self.stop_list[s] in self.user_response:
            #         self.close_conversation()  # end conversation and ask for rating
            #         self.conversation.initialize_conversation()
            #         break

            if dialogflow_reply.response.query_result.intent:
                intent_name = dialogflow_reply.response.query_result.intent.display_name  # get intent name
                print(f"INTENT '{intent_name}'")
                if intent_name == 'stop_conversation':
                    # self.close_conversation()  # end conversation and ask for rating
                    print('END CONVERSATION INTENT')
                    # self.conversation.initialize_conversation()
                    self.interaction.append('Conversation Complete: User ended interaction')

        # if self.interaction:
        save_list_to_file(self.interaction, 'Interaction_files/interaction_text_05_07_test.txt')
        self.interaction.clear()  # clear interaction list
        # self.close_conversation()  # end conversation and ask for rating
        print('CONVERSATION ENDED')
        self.conversation.initialize_conversation()  # initialize conversation to original state


if __name__ == '__main__':
    GPTPepper().run()
