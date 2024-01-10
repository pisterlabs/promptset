# Load your env variables
from dotenv import load_dotenv
load_dotenv()

# Import your dependencies
import dearpygui.dearpygui as dpg
from nylas import APIClient
import os
import re
import openai
import speech_recognition as sr
import boto3
import time
import jellyfish
import textwrap
from playsound import playsound

# Some global variables
emails = []
title = ""
signature = "\nBlag aka Alvaro Tejada Galindo\nSenior Developer Advocate, Nylas"
names_list = {'name': ""}
email_list = {'email': ""}
contacts_limit = 10
openai.api_key = os.environ.get("OPEN_AI")

# Play the audio we recorded with text-to-speech
def play_audio(audio_file):
    playsound(audio_file)

# Pass the text and generate the text-to-speech as a file
def synthesize_speech(text, output_filename):
    polly = boto3.client('polly', region_name='us-west-2')
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Joanna',
        Engine='neural'
    )

    with open(output_filename, 'wb') as f:
        f.write(response['AudioStream'].read())

# Use speech recognition to get input from the microphone
# And return the input as text
def record_input(output_filename = ""):
    with sr.Microphone() as source:
        recognizer = sr.Recognizer()
        source.pause_threshold = 1
        audio = recognizer.listen(source, phrase_time_limit = None, timeout = None)
        try:
            transcription = recognizer.recognize_google(audio)
        except:
            transcription = ""
            synthesize_speech("Sorry, I didn't get that. Please start again.", 'response.mp3')
            play_audio('response.mp3')
    return transcription

# Grab the audio input and produce text
def transcribe_audio_to_text(filename):
    recogizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recogizer.record(source) 
    try:
        return recogizer.recognize_google(audio)
    except:
        speak_text("There was an error")

# To define the color of the buttons
def _hsv_to_rgb(h, s, v):
    if s == 0.0: return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
    if i == 0: return (255*v, 255*t, 255*p)
    if i == 1: return (255*q, 255*v, 255*p)
    if i == 2: return (255*p, 255*v, 255*t)
    if i == 3: return (255*p, 255*q, 255*v)
    if i == 4: return (255*t, 255*p, 255*v)
    if i == 5: return (255*v, 255*p, 255*q)

# Create the GUI Context
dpg.create_context()

# Register our font and size
with dpg.font_registry():
    default_font = dpg.add_font("fonts/Roboto-Regular.ttf", 30)
    
# Display an information pop-up message
def show_info(title, message, selection_callback):
    with dpg.mutex():
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()

        with dpg.window(label=title, modal=True, no_close=True) as modal_id:
            dpg.add_text(message)
            dpg.add_button(label="Ok", width=75, user_data=(modal_id, True), callback=selection_callback)

    dpg.split_frame()
    width = dpg.get_item_width(modal_id)
    height = dpg.get_item_height(modal_id)
    dpg.set_item_pos(modal_id, [viewport_width // 2 - width // 2, viewport_height // 2 - height // 2])

# Close the message box
def on_selection(sender, unused, user_data):
    dpg.delete_item(user_data[0])

# Initialize an instance of the Nylas SDK using the client credentials
nylas = APIClient(
    os.environ.get("CLIENT_ID"),
    os.environ.get("CLIENT_SECRET"),
    os.environ.get("ACCESS_TOKEN")
)

# Load contacts using the contacts endpoint
def load_config(emails):
    contacts = nylas.contacts.where(source = 'address_book', limit = contacts_limit)
    for contact in contacts:
        email_detail = contact.given_name + " " + contact.surname + ": "
        try:
            email_detail += contact.emails["personal"][0]
            emails.append(email_detail)
        except:
            try:
                email_detail += contact.emails["work"][0]
                emails.append(email_detail)
            except:
                email_detail += contact.emails[None][0]
                emails.append(email_detail)
    return emails

# Calls the Email Wizard: Voice recognition, text-to-speech, ChatGPT
def wizard_callback(sender, app_data, user_data):
    emails = []
    load_config(emails)
    synthesize_speech("Welcome to Nylas Email Wizard. Who are you sending an email?", 'response.mp3')
    play_audio('response.mp3')
    transcription = record_input("")
    if transcription != "":
        # We can say more than one name using an "and"
        split_transcription = transcription.split(" and ")
        for name in split_transcription:
            for found_email in emails:
				# Search for requested people in contacts emails
                full_name = re.search("^.*?(?=:)", found_email)
                first_name = re.search("^\S+", full_name.group())
                # Use soundex to match names that are not exactly the same
                if jellyfish.soundex(name) == jellyfish.soundex(first_name.group()):
                    set_emails(found_email, txtTo)
                else:
                    if jellyfish.soundex(name[0:3]) == jellyfish.soundex(first_name.group()):
                       set_emails(found_email, txtTo)
		
        transcription_emails = transcription
        synthesize_speech(f"Adding {transcription} to the list of recipients", 'response.mp3')
        play_audio('response.mp3')
        # Ask for title
        synthesize_speech("Let me know the title", 'response.mp3')
        play_audio('response.mp3')
        transcription = record_input("")
        synthesize_speech(f"Adding {transcription} to the title", 'response.mp3')
        play_audio('response.mp3')
        dpg.set_value(txtTitle, transcription.capitalize())
        # Ask for email content
        synthesize_speech("Let me know the content of the email", 'response.mp3')
        play_audio('response.mp3')
        transcription = record_input("")
        # Prompt for ChatGPT					
        prompt = """
        write a short email to Name with no subject to talk about Content
        
        email:
        """     
        # Replace Name and Content with proper information
        prompt = re.sub("Name", transcription_emails, prompt)
        prompt = re.sub("Content", transcription, prompt)
        synthesize_speech(f"I'm generating the content of the email. Please wait.", 'response.mp3')
        play_audio('response.mp3')        
        # Call ChatGPT
        response = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=100, temperature=0)
        # Replace signature on reponse
        wrapper = textwrap.TextWrapper(width=75)
        body = response["choices"][0]["text"]
        word_list = wrapper.wrap(text=body)
        body = ""
        for word in word_list:
            body  = body + word + "\n"
        body = re.sub("(\[Your Name\])", signature, body)
        # Add ChatGPT response
        dpg.set_value(txtBody, body)
        synthesize_speech(f"I added the content to the email", 'response.mp3')
        play_audio('response.mp3')
        # Wait for one second
        time.sleep(1)
        synthesize_speech("Would you like to send the email now?", 'response.mp3')
        play_audio('response.mp3')
        transcription = record_input("")
        # If we answer yes, then send the email
        if transcription.lower() != "no":
            body = response["choices"][0]["text"]
            body = re.sub("(\[Your Name\])", signature, body)
            emails = dpg.get_value(txtTo).rstrip(";")
            title = dpg.get_value(txtTitle)
            body = re.sub('\n', '<br>', body)
            send_email(emails, title, body, [txtTo, txtTitle, txtBody])
            synthesize_speech("The email was sent", 'response.mp3')
            play_audio('response.mp3')

def send_email(emails, title, body, user_data):
    draft = nylas.drafts.create()
    participants = []
    draft.subject = title
    draft.body = body
    list_of_emails = email_list['email'].rstrip(";").split(";")
    list_of_names = names_list['name'].rstrip(";").split(";")
    for i in range(0, len(list_of_emails)):
        participants.append({"name":list_of_names[i],"email":list_of_emails[i]})
    draft.to = participants
    draft.send()
    email_list['email'] = ""
    dpg.set_value(user_data[0], email_list['email'])
    names_list['name'] = ""
    dpg.set_value(user_data[1],"")
    dpg.set_value(user_data[2],"")
    show_info("Message Box", "The email was sent successfuly", on_selection)

# Set the emails on the email field
def set_emails(email_info, user_data):
    email = re.search("(?<=: ).*$", email_info)
    email_list['email'] = email_list['email'] + email.group() + ";"
    dpg.set_value(user_data, email_list['email'])
    name = re.search("^.*?(?=:)", email_info)
    names_list['name'] = names_list['name'] + name.group() + ";"	

# When we select a contact from the contact list, add it
def contact_callback(sender, app_data, user_data):
    set_emails(app_data, user_data)
	
# Clear the email field
def clear_callback(sender, app_data, user_data):
    email_list['email'] = ""
    dpg.set_value(user_data, email_list['email'])
    names_list['name'] = ""

# We pressed the send button
def send_callback(sender, app_data, user_data):
    emails = dpg.get_value(txtTo).rstrip(";")
    title = dpg.get_value(txtTitle)
    body = re.sub('\n', '<br>', dpg.get_value(txtBody))
    send_email(emails, title, body, user_data)
	
# When we press cancel, we clear all fields
def cancel_callback(sender, app_data, user_data):
    email_list['email'] = ""
    dpg.set_value(user_data[0], email_list['email'])
    names_list['name'] = ""
    dpg.set_value(user_data[1], "")
    dpg.set_value(user_data[2], "")

# Create the main window
with dpg.window(label="", tag="window", pos=(0,0), width=1100, height=420, no_resize = True, no_close = True, no_collapse = True):
	# Bind font with all application components
    dpg.bind_font(default_font)
    
    # Load contacts
    load_config(emails)
    # Add the Email Wizard button
    btnWizard = dpg.add_button(label="Nylas Email Wizard", callback=wizard_callback)
    # Add a space between components
    dpg.add_spacer()
    # Define a group
    group = dpg.add_group(horizontal=True)
    # Add To field
    txtTo = dpg.add_input_text(label="To", width = 500, parent = group)
    # Add contacts combobox
    cboContacts = dpg.add_combo(emails, label = "Contacts", width = 300, callback=contact_callback, parent = group, user_data = txtTo)
    # Add Clear button
    btnClear = dpg.add_button(label="Clear", callback=clear_callback, parent = group, user_data = txtTo)
    # Add Title field
    txtTitle = dpg.add_input_text(label="Title", width = 842, user_data = txtTo)
    # Add Body field
    txtBody = dpg.add_input_text(label="Body", multiline=True, width=842, height = 200, tab_input=True)
    # Define a new group
    buttons_group = dpg.add_group(horizontal=True)
    # Add Send and Cancel buttons
    btnSend = dpg.add_button(label="Send", callback=send_callback, parent = buttons_group, user_data = [txtTo, txtTitle, txtBody])
    btnCancel = dpg.add_button(label="Cancel", callback=cancel_callback, parent = buttons_group, user_data = [txtTo, txtTitle, txtBody])

# Specify the theme for the Email wizard button
with dpg.theme() as wizard_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(1/7.0, 0.6, 0.6))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0, category=dpg.mvThemeCat_Core)

# Specify the theme for the all remaining buttons        
with dpg.theme() as button_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(4/7.0, 0.6, 0.6))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0, category=dpg.mvThemeCat_Core)        

# Bind themes with buttons
dpg.bind_item_theme(btnWizard, wizard_theme)
dpg.bind_item_theme(btnSend, button_theme)
dpg.bind_item_theme(btnCancel, button_theme)
dpg.bind_item_theme(btnClear, button_theme)

# Define the application viewport
dpg.create_viewport(title='Email Assistant', width=1100, height=420)
# Initialize the application
dpg.setup_dearpygui()
# Display the viewport
dpg.show_viewport()
# Start our application
dpg.start_dearpygui()
# Destroy the context when the application ends
dpg.destroy_context()
