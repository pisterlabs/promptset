import os, logging, uuid, requests, re, base64
from pymongo import MongoClient
from openai import OpenAI
from pygwan import WhatsApp
from PIL import Image
from .generics import get_recipient_chat_history


token = os.environ.get("WHATSAPP_ACCESS_TOKEN")
phone_number_id = os.environ.get("PHONE_NUMBER_ID")
openai_api_key = str(os.environ.get("OPENAI_API_KEY"))
messenger = WhatsApp(token=token, phone_number_id=phone_number_id)
oai = OpenAI(api_key=openai_api_key)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

######################################## users database functions ########################################
def recipients_database():
    """users' database connection object"""
    client = MongoClient(
        "mongodb://mongo:xQxzXZEzUilnKKhrbELE@containers-us-west-114.railway.app:6200"
    )
    database = client["users"]
    collection = database["recipients"]
    return collection

def check_id_database(message_stamp: str):
    """Check if a message_stamp(combination of conersation_id+message_id) is in the database or not."""
    # users' database connection object
    client = MongoClient(
        "mongodb://mongo:xQxzXZEzUilnKKhrbELE@containers-us-west-114.railway.app:6200"
    )
    database = client["Readit"]
    collection = database["messageids"]

    # Query the collection to check if the message_id exists
    query = {"message_stamp": message_stamp}
    result = collection.find_one(query)

    # Close the database connection
    client.close()

    # If the result is not None, the message_id exists in the database
    return result is not None

def add_id_to_database(message_stamp: str):
    """Add a message_stamp to the database."""
    # users' database connection object
    client = MongoClient(
        "mongodb://mongo:xQxzXZEzUilnKKhrbELE@containers-us-west-114.railway.app:6200"
    )
    database = client["Readit"]
    collection = database["messageids"]
    document = {"message_stamp": message_stamp}
    collection.insert_one(document)
    client.close()  
 
######################################### thread functions ##############################################
    
def save_thread_id(thread_id : str, recipient):
    """saves a user's thread id in the MongoDB database."""
    try:
        client = MongoClient("mongodb://mongo:xQxzXZEzUilnKKhrbELE@containers-us-west-114.railway.app:6200")
        database = client["users"]
        collection = database["threads"]
        query = {"key": recipient}
        new_values = {"$set": {"thread_id": thread_id}}
        result = collection.update_one(query, new_values, upsert=True)
        client.close()
        if result.modified_count > 0 or result.upserted_id is not None:
            logging.info("===================================SAVED THREAD_ID: %s", thread_id)
            return "success"
        else:
            return "failed"
    except Exception as e:
        print(f"An error occurred: {e}")
        return "failed"

def get_thread_id(recipient):
    """Fetches the recipient's thread id from the MongoDB database."""
    client = MongoClient("mongodb://mongo:xQxzXZEzUilnKKhrbELE@containers-us-west-114.railway.app:6200")
    database = client["users"]
    collection = database["threads"]
    query = {"key": recipient}
    result = collection.find_one(query)
    if result and 'thread_id' in result:
        logging.info("===================================FETCHED THREAD_ID: %s", result['thread_id'])
        return str(result['thread_id'])
    else:
        return "no thread found" # Or a default rate if not found
    
########################################## miscellenous functions ########################################    
    
def language_check(transcript: str):
    '''This function checks if the argument is english which makes sense or not'''
    content = f"""Classes: [`sensible`, `non-sensible`]
    Text: {transcript}
    classify the text into one of the above classes.
    If you think the text is sensible, type `sensible` otherwise type `non-sensible`. ONLY TYPE THE WORDS IN THE QUOTES."""
    completion = oai.chat.completions.create(
    model="gpt-3.5-turbo",
    temperature=0.6,
    messages=[
        {"role": "user", "content": content},
    ]
    )
    if completion.choices[0].message.content == 'sensible':
        return True
    elif completion.choices[0].message.content == 'non-sensible':
        return False
    else:
        logging.error(f"===================================ERROR IN DETERMINING LEGIBILITY OF THE ENGLISH: {completion.choices[0].message.content}")
        return completion.choices[0].message.content
    
def encode_image(image_path):
    '''This function encodes an image into base64'''
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def link_removal(response: str):
    '''this function takes in the response from the assistant and removes the links from it'''
    image_pattern = r"https?://(?:[a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,6}(?:/[^/#?]+)+\.(?:png|jpe?g|gif|webp|bmp|tiff|svg)"
    reply_without_links = re.sub(image_pattern, "", response)
    colon_index = reply_without_links.find(":")
    if colon_index != -1:
        reply_without_links = reply_without_links[:colon_index]
        reply_without_links = reply_without_links.strip()
    return reply_without_links

def response_handler(response: str, recipient_id: str, message_id: str):
    history = get_recipient_chat_history(recipient=recipient_id)
    '''this function takes in the response from the assistant, checks if it contains a link,
    if it does, it extracts the link and sends the image to the user,
    if it doesn't, it sends the response to the user'''
    reply_without_links = link_removal(response=response)
    url_match = re.search(r"!\[.*?\]\((https.*?)\)", response)
    if url_match:
        extracted_url = url_match.group(1)
        logging.info("==================================================== EXTRACTED URL: %s", extracted_url)
        r = requests.get(extracted_url, allow_redirects=True)
        image_name = f'{uuid.uuid4()}.png'
        with open(image_name, 'wb') as f:
            f.write(r.content)
            f.close()
            logging.info(f"==================================================== SAVED IMAGE AS: {image_name}")
        try:
            new_image_name = f'{uuid.uuid4()}.jpeg'
            with Image.open((os.path.realpath(image_name))) as img:
                rgb_im = img.convert('RGB')  # Convert to RGB
                rgb_im.save(new_image_name, 'JPEG', quality=90)  # Save as JPEG with quality 90
            image_id_dict = messenger.upload_media(media=(os.path.realpath(new_image_name)))
            messenger.send_image(
            image=image_id_dict["id"],
            recipient_id=recipient_id,
            caption=reply_without_links,
            link=False,)
            # Delete the image from the server
            os.remove(path=(os.path.realpath(image_name)))
            os.remove(path=(os.path.realpath(new_image_name)))
        except IOError as e:
            logging.error(f"==================================================== ERROR OCCURED: {e}")
            messenger.send_message(message=f"Error occured: {e}", recipient_id=recipient_id)
            history.add_ai_message(message=f"Error occured: {e}")
        except Exception as e:
            logging.error(f"==================================================== ERROR OCCURED: {e}")
            messenger.send_message(message=f"Error occured: {e}", recipient_id=recipient_id)  
    else:
        messenger.reply_to_message(message_id=message_id, recipient_id=recipient_id, message=response)
        history.add_ai_message(message=response)
        
def audio_response_handler(response: str, recipient_id: str, ai, message_id=None,):
    '''this function takes in the response from the assistant, checks if it contains a link,
    if it does, it extracts the link and sends the image to the user,
    if it doesn't, it creates and sends the audio response to the user'''
    history = get_recipient_chat_history(recipient=recipient_id)
    reply_without_links = link_removal(response=response)
    url_match = re.search(r"!\[.*?\]\((https.*?)\)", response)
    if url_match:
        extracted_url = url_match.group(1)
        logging.info("==================================================== EXTRACTED URL: %s", extracted_url)
        r = requests.get(extracted_url, allow_redirects=True)
        image_name = f'{uuid.uuid4()}.png'
        with open(image_name, 'wb') as f:
            f.write(r.content)
            f.close()
            logging.info(f"==================================================== SAVED IMAGE AS: {image_name}")
        try:
            new_image_name = f'{uuid.uuid4()}.jpeg'
            with Image.open((os.path.realpath(image_name))) as img:
                rgb_im = img.convert('RGB')  # Convert to RGB
                rgb_im.save(new_image_name, 'JPEG', quality=90)  # Save as JPEG with quality 90
            image_id_dict = messenger.upload_media(media=(os.path.realpath(new_image_name)))
            messenger.send_image(
            image=image_id_dict["id"],
            recipient_id=recipient_id,
            caption=reply_without_links,
            link=False,)
            # Delete the image from the server
            os.remove(path=(os.path.realpath(image_name)))
            os.remove(path=(os.path.realpath(new_image_name)))
        except IOError as e:
            logging.error(f"==================================================== ERROR OCCURED: {e}")
            messenger.send_message(message=f"Error occured: {e}", recipient_id=recipient_id)
        except Exception as e:
            logging.error(f"==================================================== ERROR OCCURED: {e}")
            messenger.send_message(message=f"Error occured: {e}", recipient_id=recipient_id)  
    else:
        audio = ai.create_audio(script=response)
        audio_id_dict = messenger.upload_media(media=(os.path.realpath(audio)))
        messenger.send_audio(audio=audio_id_dict["id"], recipient_id=recipient_id, link=False)
        history.add_ai_message(message=response)
        

    