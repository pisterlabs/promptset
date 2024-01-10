import sqlite3
import subprocess
import time
import os
import re
from openai import OpenAI
import openai
import base64
import requests
import getpass
from PIL import Image
from pydub import AudioSegment
import tempfile

DB_PATH = f"/Users/{getpass.getuser()}/Library/Messages/chat.db" #path to chat.db file

# function: gets contact number from contact name
# parameters: name - string
# returns: phone number
def get_contact_number(name):
    script = f'osascript getContactNumber.applescript "{name}"' #define applescript to get contact number
    process = subprocess.Popen(script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) #run applescript
    output, error = process.communicate() #get output and error from applescript

    if process.returncode == 0: #if applescript ran successfully
        phone_number = output.decode('utf-8').strip() #get phone number from output
        filtered_number = re.sub(r"[^\d+]", "", phone_number) #filter out non-numeric characters
        return filtered_number #return filtered number
    else:
        raise Exception("Error: " + error.decode('utf-8')) #raise exception if applescript failed
    
# function: gets the id of the last message in the database
# parameters: none
# returns: id of last message
def get_last_message_id():
    conn = sqlite3.connect(DB_PATH) #connect to database
    cursor = conn.cursor() #create cursor
    cursor.execute("SELECT ROWID FROM message ORDER BY ROWID DESC LIMIT 1") #get id of last message
    last_row_id = cursor.fetchone()[0] #get id from cursor
    conn.close() #close connection
    return last_row_id #return id

# function: get recent messages from a specific contact
# parameters: target_number - string, last_id_checked - int
# returns: list of messages
def get_recent_messages(target_number, last_id_checked, output_buffer):
    conn = sqlite3.connect(DB_PATH) #connect to database
    cursor = conn.cursor() #create cursor

    cursor.execute(""" 
        SELECT m.ROWID, m.text, CASE WHEN a.filename IS NOT NULL THEN 1 ELSE 0 END as is_media, a.mime_type, a.filename
        FROM message m
        LEFT JOIN message_attachment_join maj ON m.ROWID = maj.message_id
        LEFT JOIN attachment a ON maj.attachment_id = a.ROWID
        INNER JOIN handle h ON m.handle_id = h.ROWID
        WHERE m.ROWID > ? AND h.id = ? AND m.is_from_me = 0
        ORDER BY m.ROWID DESC
        """, (last_id_checked, target_number))  # Get recent messages from target number

    messages = cursor.fetchall() #get messages from cursor
    conn.close() #close connection

    processed_messages = postprocess_messages(messages, output_buffer) #postprocess messages
    return processed_messages #return messages

# function: deal with attatchments and reactions
# parameters: target_number - string, last_id_checked - int, pattern - string
# returns: list of messages
def postprocess_messages(messages, output_buffer):
    pattern = r'^(Loved|Liked|Disliked|Laughed at|Emphasized) “.*”$' #define pattern to check for reactions
    processed_messages = [] #create messages list
    for message in messages: #for each message in recent messages
        message_id, text, is_media, file_type, filepath = message
        text = text.replace('[', '').replace(']', '').replace('<', '').replace('>', '') #replace brackets and angle brackets
        is_media = bool(is_media)
        if not re.match(pattern, text): #if message is not a reaction
            if is_media: #if message is media
                filepath = filepath.replace('~', f'/Users/{getpass.getuser()}') #replace ~ with user directory
                if file_type:
                    if file_type.startswith("image"): #if message is an image
                        processed_messages.append((message_id, generate_image_description(filepath, output_buffer), is_media, file_type, filepath)) #add image description to messages
                    elif file_type.startswith("video"):
                        processed_messages.append((message_id, "Imagine you've received a video message from a friend, but you're currently unable to watch it. Craft a polite and believable excuse explaining why you can't watch the video right now.", is_media, file_type, filepath)) #add excuse for video to messages
                elif filepath.endswith(".caf"):
                    processed_messages.append((message_id, generate_audio_transcript(filepath, output_buffer), is_media, file_type, filepath)) #add audio description to messages
            else: #if message is not media
                processed_messages.append((message_id, text, is_media, file_type, filepath)) #add message to messages
    return processed_messages #return messages

# function: generate a response from GPT
# parameters: incoming_message - string, conversation_history - list of messages, recipient_name - string, recipient_description - string
# returns: response message
def generate_response(incoming_message, conversation_history, user_name, recipient_name, recipient_description, conversation_context, gpt_model, contains_images, output_buffer):
    client = OpenAI() #create OpenAI client

    # Construct the initial part of the content
    if conversation_context:
        context_part = f"Additional context for this situation is provided as: {conversation_context}."
    else:
        context_part = "No additional context is provided for this situation."
    
    # Check if the message contains images
    if contains_images:
        image_response_part = "Since the message contains an image, provide a brief and concise reaction to the image as if you were seeing it yourself."
    else:
        image_response_part = (
            "It's important to note that the message does not contain an actual image. Therefore, treat any descriptions or mentions of images as plain text and respond accordingly. "
            "Do not react as if viewing an actual image, but rather focus on the textual content of the message for your response."
        )
    # Construct the final content
    content = (
        f"As an imaginative person named {user_name}, you're about to respond to a text from {recipient_name}, your {recipient_description}. "
        f"{context_part} This context is key to fully grasping the subtleties of your interaction with {recipient_name}. "
        f"{image_response_part} Keep this in mind while responding. Your replies should seamlessly mirror those of a real person. "
        "If a task is asked of you that cannot be accomplished via text, politely decline with a creative, yet plausible excuse. "
        "However, if the task is doable through text, respond appropriately without creating an excuse. "
        "Consistently maintain the persona of a real individual, adapting your responses to the unique situation and the given context."
    )
        
    messages = [{ #give GPT inital instruction prompt
        "role": "assistant", 
        "content": content}]
    messages.extend(conversation_history)  #add previous conversation

    pattern = r'^.*Questioned “(.*?)”.*$' #define pattern to check for questioned text
    questioned_text = re.search(pattern, incoming_message) #search for questioned text
    if questioned_text: #if questioned text is found
        questioned_text = questioned_text.group(1) #get questioned text
        output_buffer.append(f"questioned_text: {questioned_text}\n")
        incoming_message = re.sub(pattern, f"I don't understand your previous text... {questioned_text}. Can you please provide more information?", incoming_message) #rephrase incoming message
    
    messages.append({"role": "user", "content": incoming_message}) #add incoming message

    try:
        completion = client.chat.completions.create( #generate response from GPT
            model=gpt_model, #use gpt_model
            messages=messages #use compiled messages as prompt
        )
    except openai.BadRequestError as e:
        output_buffer.append(f"OpenAI API error: {e}\n")
        return None

    response = completion.choices[0].message.content #get response from GPT

    if "AI" in response: #if response contains AI
        output_buffer.append("AI detected in response. Rephrasing...\n")
        new_message = f"Your message ({response}) refers to AI. Please remember to maintain the persona of a human responder in our conversation. If you're indicating that you are an AI, kindly rephrase your message to exclude this information." #rephrase response
        messages.append({"role": "assistant", "content": new_message}) #add rephrased response to messages
        try:
            completion = client.chat.completions.create( #generate response from GPT
                model=gpt_model, #use gpt_model
                messages=messages #use compiled messages as prompt
            )
        except openai.BadRequestError as e:
            output_buffer.append(f"OpenAI API error: {e}\n")
            return None
        response = completion.choices[0].message.content #get response from GPT

    conversation_history.append({"role": "user", "content": incoming_message}) #add incoming message to conversation history
    conversation_history.append({"role": "assistant", "content": response}) #add response to conversation history

    return response #return response

# function: get the wait time for a response
# parameters: message - string
# returns: wait time in seconds
def get_response_time(message, words_per_minute):
    word_count = len(message.split()) #get word count of message
    response_time = word_count * 60/words_per_minute #calculate response time
    return response_time #return response time
    # return 0 #return 0 for testing purposes

# function: convert image using imagemagick
# parameters: input_path - string, output_path - string
# returns: output_path
def convert_image_with_imagemagick(input_path, output_path):
    subprocess.run(['magick', 'convert', input_path, output_path], check=True) #convert image using imagemagick
    return output_path #return output path

# function: convert image using Pillow
# parameters: input_path - string, output_path - string, format - string
# returns: output_path
def convert_image(input_path, output_path, format='JPEG'):
    try: #try to convert image using Pillow
        with Image.open(input_path) as img: #open image
            img.convert('RGB').save(output_path, format) #convert image
        return output_path #return output path
    except IOError: #if Pillow fails to convert image
        return convert_image_with_imagemagick(input_path, output_path) #convert image using imagemagick
    
# function: encode image to base64
# parameters: filepath - string
# returns: base64 image
def encode_image_to_base64(filepath):
    supported_formats = ['png', 'jpeg', 'gif', 'webp'] #define supported formats

    file_ext = filepath.split('.')[-1].lower() #get file extension
    if file_ext not in supported_formats: #if file extension is not supported
        filepath = convert_image(filepath, 'temp_converted_image.jpeg') #convert image to jpeg

    with open(filepath, "rb") as image_file: #open image file
        base64_image = base64.b64encode(image_file.read()).decode('utf-8') #encode image to base64

    if filepath == 'temp_converted_image.jpeg': #if image was converted
        os.remove(filepath) #remove converted image

    return base64_image #return base64 image

# function: generate a description for the inputted image
# parameters: message - string
# returns: wait time in seconds
def generate_image_description(filepath, output_buffer):
    api_key = os.getenv('OPENAI_API_KEY') #get OpenAI API key

    base64_image = encode_image_to_base64(filepath) #encode image to base64

    headers = {
        "Content-Type": "application/json", #set content type to json
        "Authorization": f"Bearer {api_key}" #set authorization to api key
    }

    payload = {
        "model": "gpt-4-vision-preview", #use gpt-4-vision-preview model
        "messages": [{
            "role": "user",
            "content": [{ 
                "type": "text", 
                "text": "Please provide a detailed description of the uploaded image, including its setting, main subjects, notable objects, mood, and other key elements."}, #give GPT inital instruction prompt
            {"type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}" #set image url to base64 image
            }
            }
            ]
        }],
        "max_tokens": 1200 #set max tokens
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) #generate response from GPT
    except Exception as e:
        output_buffer.append(f"OpenAI API error: {e}\n")
        return None

    response_data = response.json() #get json response data
    image_desciption = response_data['choices'][0]['message']['content'] #get message content from response data
    output_buffer.append(f"image_desciption: {image_desciption}\n")
    formatted_image_description = "Imagine you are directly looking at an image described as follows: '" + image_desciption.replace('\n', ' ') + "'. Please provide a brief and concise reaction to the image as if you were seeing it yourself, keeping your response short."

    return formatted_image_description #return formatted image description

# function: generate a transcript for the inputted audio
# parameters: filepath - string, output_buffer - list
# returns: transcript text
def generate_audio_transcript(filepath, output_buffer):
    client = openai.Client()  # Initialize the OpenAI client

    # Convert the CAF file to MP3 and save it to a temporary file
    with open(filepath, "rb") as audio_file:
        caf_audio = AudioSegment.from_file(audio_file, format="caf")
        
        # Use a temporary file to store the MP3 data
        temp_mp3_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        caf_audio.export(temp_mp3_file.name, format="mp3")
        temp_mp3_file.close()

    # Call the OpenAI API with the path to the temporary MP3 file
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(temp_mp3_file.name, "rb")
        )
        transcript_text = transcript.text
        output_buffer.append(f"transcript_text: {transcript_text}\n")
        return transcript_text
    except openai.BadRequestError as e:
        output_buffer.append(f"OpenAI API error: {e}\n")
        return None
    finally:
        os.remove(temp_mp3_file.name)
    
# function: sleep for a given amount of time or until a stop flag is set
# parameters: sleep_time - int, stop_flag - threading.Event
# returns: nothing
def sleep_with_check(sleep_time, stop_flag):
    elapsed_time = 0 #set elapsed time to 0
    while elapsed_time < sleep_time and not stop_flag.is_set(): #while elapsed time is less than sleep time and stop flag is not set
        time.sleep(min(1, sleep_time - elapsed_time)) #sleep for 1 second or until sleep time is reached
        elapsed_time += 1 #increment elapsed time

def check_for_images(messages):
    for message in messages:
        if message[2] and message[3] and message[3].startswith("image"):
            return True
    return False

# function: have a conversation with AI using a target name
# parameters: target_name - string, target_description - string
# returns: nothing
def converse_with_AI(target_number, target_name, user_name, target_description, words_per_minute, conversation_context, gpt_model, stop_flag, output_buffer):
    CONVERSATION_HISTORY = [] #create conversation history
    output_buffer.append(f"listening for messages from {target_number}\n")
    last_id_checked = get_last_message_id() #get id of last message
    check_interval = 5 #set check interval
    while not stop_flag.is_set(): #loop until stop flag is set
        start_time = time.time() #get start time
        messages = get_recent_messages(target_number, last_id_checked, output_buffer) #get recent messages
        contains_images = check_for_images(messages)
        if len(messages) > 0: #if there are new messages
            concatenated_text = ' '.join([row[1] for row in messages[::-1]]) #concatenate messages
            output_buffer.append(f"concatenated_text: {concatenated_text}\n")

            output_buffer.append("generating ai response...\n")
            response_message = generate_response(concatenated_text, CONVERSATION_HISTORY, user_name, target_name, target_description, conversation_context, gpt_model, contains_images, output_buffer) #generate response
            output_buffer.append(f"response_message: {response_message}\n")

            response_generation_time = time.time() - start_time #calculate response generation time
            output_buffer.append(f"response_generation_time: {response_generation_time}")
            response_time = get_response_time(response_message, words_per_minute) #get response time
            output_buffer.append(f"response_time: {response_time}")
            wait_time = response_time - response_generation_time #get wait time
            if wait_time < 0: #if response time is negative
                wait_time = 0
            total_time_waited = wait_time + response_generation_time#set total time waited
            output_buffer.append(f"Sleeping for {wait_time} seconds\n")
            sleep_with_check(wait_time, stop_flag) #sleep for response time if stop flag is not set

            if stop_flag.is_set():
                break
            output_buffer.append("checking for new messages...\n")
            start_time = time.time() #get start time
            new_messages = get_recent_messages(target_number, last_id_checked, output_buffer) #get recent messages
            contains_images = check_for_images(messages)
            while len(new_messages) > len(messages): #while there are new messages
                output_buffer.append("new message received\n")
                messages = new_messages #update messages
                concatenated_text = ' '.join([row[1] for row in messages[::-1]]) #concatenate messages
                output_buffer.append(f"new concatenated_text: {concatenated_text}\n")

                output_buffer.append("generating new ai response...\n")
                response_message = generate_response(concatenated_text, CONVERSATION_HISTORY, user_name, target_name, target_description, conversation_context, gpt_model, contains_images, output_buffer) #generate response
                output_buffer.append(f"new response_message: {response_message}\n")
                
                if stop_flag.is_set(): #if stop flag is set
                    break #break out of loop

                response_generation_time = time.time() - start_time #calculate response generation time
                output_buffer.append(f"response_generation_time: {response_generation_time}")
                response_time = get_response_time(response_message, words_per_minute) #get response time
                output_buffer.append(f"response_time: {response_time}")
                wait_time = response_time - response_generation_time #get response time
                if wait_time < 0: #if response time is negative
                    wait_time = 0 #wait for 0 seconds
                output_buffer.append(f"already waited {total_time_waited} seconds")
                remaining_wait_time = wait_time - total_time_waited #calculate wait time
                if remaining_wait_time < 0: #if wait time is negative
                    remaining_wait_time = 0 #wait for 0 seconds
                total_time_waited += remaining_wait_time + response_generation_time #update total time waited
                output_buffer.append(f"Sleeping for {remaining_wait_time} seconds\n")
                sleep_with_check(remaining_wait_time, stop_flag) #sleep for response time if stop flag is not set

                if stop_flag.is_set():
                    break
                output_buffer.append("checking for new messages...\n")
                start_time = time.time() #get start time
                new_messages = get_recent_messages(target_number, last_id_checked, output_buffer) #get recent messages
                contains_images = check_for_images(messages)
            output_buffer.append(f"sending message\n")
            escaped_response_message = response_message.replace('"', '\\"')
            os.system(f'osascript sendMessage.applescript "{target_number}" "{escaped_response_message}"')            
            last_id_checked = messages[0][0] #update last id checked

        sleep_with_check(check_interval, stop_flag) #sleep for check interval if stop flag is not set

# if __name__ == "__main__":
