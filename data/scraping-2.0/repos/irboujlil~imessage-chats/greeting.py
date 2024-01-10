from flask import Flask, request, jsonify
#from sendblue import Sendblue
import requests
import sqlite3
import datetime
import json
import openai
import os

openai.api_key = os.environ.get('API_TOKEN')
SENDBLUE_API_KEY = os.environ.get('SENDBLUE_API_KEY')
SENDBLUE_API_SECRET = os.environ.get('SENDBLUE_API_SECRET')
#sendblue = Sendblue(SENDBLUE_API_KEY, SENDBLUE_API_SECRET)
app = Flask(__name__)

def get_chat_mapping(chatdb_location):
    conn = sqlite3.connect(chatdb_location)
    cursor = conn.cursor()

    cursor.execute("SELECT room_name, display_name FROM chat")
    result_set = cursor.fetchall()

    mapping = {room_name: display_name for room_name, display_name in result_set}

    conn.close()

    return mapping

# Function to read messages from a sqlite database
def read_messages(chatdb_location, n, self_number='Me', human_readable_date=True):
    # Connect to the database and execute a query to join message and handle tables
    conn = sqlite3.connect(chatdb_location)
    cursor = conn.cursor()
    query = """
    SELECT message.ROWID, message.date, message.text, message.attributedBody, handle.id, message.is_from_me, message.cache_roomnames
    FROM message
    LEFT JOIN handle ON message.handle_id = handle.ROWID
    """
    if n is not None:
        query += f" ORDER BY message.date DESC LIMIT {n}"
    results = cursor.execute(query).fetchall()
    
    # Initialize an empty list for messages
    messages = []

    # Loop through each result row and unpack variables
    for result in results:
        rowid, date, text, attributed_body, handle_id, is_from_me, cache_roomname = result

        # Use self_number or handle_id as phone_number depending on whether it's a self-message or not
        phone_number = self_number if handle_id is None else handle_id

        # Use text or attributed_body as body depending on whether it's a plain text or rich media message
        if text is not None:
            body = text
        
        elif attributed_body is None: 
            continue
        
        else: 
            # Decode and extract relevant information from attributed_body using string methods 
            attributed_body = attributed_body.decode('utf-8', errors='replace')
            if "NSNumber" in str(attributed_body):
                attributed_body = str(attributed_body).split("NSNumber")[0]
                if "NSString" in attributed_body:
                    attributed_body = str(attributed_body).split("NSString")[1]
                    if "NSDictionary" in attributed_body:
                        attributed_body = str(attributed_body).split("NSDictionary")[0]
                        attributed_body = attributed_body[6:-12]
                        body = attributed_body

        # Convert date from Apple epoch time to standard format using datetime module if human_readable_date is True  
        if human_readable_date:
            date_string = '2001-01-01'
            mod_date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
            unix_timestamp = int(mod_date.timestamp())*1000000000
            new_date = int((date+unix_timestamp)/1000000000)
            date = datetime.datetime.fromtimestamp(new_date).strftime("%Y-%m-%d %H:%M:%S")

        mapping = get_chat_mapping(chatdb_location)  # Get chat mapping from database location

        try:
            mapped_name = mapping[cache_roomname]
        except:
            mapped_name = None

        if(mapped_name == "Scoot FC"):
            messages.append(
            {"rowid": rowid, "date": date, "body": body, "phone_number": phone_number, "is_from_me": is_from_me,
             "cache_roomname": cache_roomname, 'group_chat_name' : mapped_name})

    conn.close()
    return messages


def print_messages(messages):
    print(json.dumps(messages))

def get_address_book(address_book_location):
    conn = sqlite3.connect(address_book_location)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT ZABCDRECORD.ZFIRSTNAME [FIRST NAME], ZABCDRECORD.ZLASTNAME [LAST NAME], ZABCDPHONENUMBER.ZFULLNUMBER [FULL NUMBER] FROM ZABCDRECORD LEFT JOIN ZABCDPHONENUMBER ON ZABCDRECORD.Z_PK = ZABCDPHONENUMBER.ZOWNER ORDER BY ZABCDRECORD.ZLASTNAME, ZABCDRECORD.ZFIRSTNAME, ZABCDPHONENUMBER.ZORDERINGINDEX ASC")
    result_set = cursor.fetchall()

    #Convert tuples to json
    json_output = json.dumps([{"FIRSTNAME": t[0], "LASTNAME": t[1], "FULLNUMBER": t[2]} for t in result_set])
    json_list = json.loads(json_output)
    conn.close()

    for obj in json_list:
        # Get the phone number from the object
        phone = obj["FULLNUMBER"]
        if phone is None:
            continue
        # Remove all non-numeric characters from the phone number
        phone = "".join([c for c in phone if c.isnumeric()])
        #if the phone number is 10 digits, add "+1" to the beginning, if it's 11 digits, add "+"
        if len(phone) == 10:
            phone = "+1" + phone
        elif len(phone) == 11:
            phone = "+" + phone
        # Add the phone number to the object
        obj["NUMBERCLEAN"] = phone
        
    new_json_output = json.dumps(json_list)
    return new_json_output

#combine recent messages and address book data
def combine_data(recent_messages, addressBookData):
    #convert addressBookData to a list of dictionaries
    addressBookData = json.loads(addressBookData)
    #loop through each message
    for message in recent_messages:
        phone_number = message["phone_number"]
        for contact in addressBookData:
            # if contact does not have property NUMBERCLEAN, skip it
            if "NUMBERCLEAN" not in contact:
                continue
            else:
                contact_number = contact["NUMBERCLEAN"]
            #if the phone number from the message matches the phone number from the contact add the names to the message
            if phone_number == contact_number:
                message["first_name"] = contact["FIRSTNAME"]
                message["last_name"] = contact["LASTNAME"]
    return 

@app.route('/print_message/', methods=['POST'])
def json_example():
    request_data = request.get_json()
    print(request_data)
    return request_data

@app.route('/getmsg/', methods=['POST'])
def respondMessage():
    request_data = request.get_json()
    print(request_data)
    #return request_data
    url = 'https://03ee-2603-7000-9200-966a-25dd-cda-2667-33e2.ngrok-free.app/msg'
    urlResponse = requests.get(url)
    urlResponse_json = urlResponse.json()
    print(urlResponse_json)
    # Retrieve the name from the url parameter /getmsg/?name=
    name = request.args.get("name", None)

    # For debugging
    print(f"Received: {name}")



    # ask the user for the location of the database
    #chatdb_location = input("Enter the absolute path of the chat database: ")
    #chatdb_location = "/Users/iliasboujlil/Library/Messages/chat.db"
    # ask the user for the location of the address book database:
    #address_book_location = input("Enter the absolute path of the address book database : ")
    #address_book_location = "/Users/iliasboujlil/Library/Application Support/AddressBook/Sources/985CDA7D-65ED-49D1-9A06-ECE5715B82AF/AddressBook-v22.abcddb"
    # ask the user for the number of messages to read
    n = 150
    #recent_messages = read_messages(chatdb_location, n)
    #print_messages(recent_messages)

    #addressBookData = get_address_book(address_book_location)
    #print(addressBookData)
    #combined_data = combine_data(recent_messages, addressBookData)
    #print_messages(combined_data)
    #messages=[{"role": "user", "content": "As an intelligent AI model, if you could be any fictional character, who would you choose and why?"}]
    message=[{"role": "user", "content": 'You are Leonardo Dicaprio. I need you to summarize the following input as if you were retelling what was going on. it is a json. You must give a narrative summary based off of first_name and body fields. Also only focus on the items with group_chat_name that includes Scoot FC. Create a narrative in the style of Leonardo Dicaprio that is not too long based off of this text. Do not start it with "In the group chat named Scoot FC and do not be cliche": {}'.format(urlResponse_json)}]
    #response = openai.Completion.create(
    #engine="text-davinci-003",
    #prompt="Summarize the text here in such a way that it is like a narrative and focus on the fields body and first_name. The field group_chat_name must also be Scoot FC:  {}".format(combined_data),
    #max_tokens=500,
    #)

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=message,
      #temperature=0,
      max_tokens=200
    )
    #print(response)

    #response = {}

    # Check if the user sent a name at all
    
    # Return the response in json format
    url = "https://api.sendblue.co/api/send-message"

    headers = {
        "sb-api-key-id": SENDBLUE_API_KEY,
        "sb-api-secret-key": SENDBLUE_API_SECRET,
        "Content-Type": "application/json",
    }

    data = {
        "number": "+15133765542",
        "content": str(response.choices[0].message.content.strip()),
        "send_style": "",
        "media_url": "",
        "status_callback": "https://03ee-2603-7000-9200-966a-25dd-cda-2667-33e2.ngrok-free.app/msg"
    }
    print("IN HERE")

    response_send_blue = requests.post(url, json=data, headers=headers)
    if response_send_blue.status_code == 200:
        print(response_send_blue.json())
        print("REQUEST SUCCESSFUL")
    else:
        print(f"Error: {response_send_blue.text}")
    print(response_send_blue)

    #response = sendblue.send_group_message(['+15133765542', '+15169967345'], response.choices[0].message.content.strip(), send_style='invisible')
    return request_data

@app.route('/getmsg/', methods=['GET'])
def respond():
    
    url = 'https://03ee-2603-7000-9200-966a-25dd-cda-2667-33e2.ngrok-free.app/msg'
    print(url);
    urlResponse = requests.get(url)
    print(urlResponse)
    urlResponse_json = urlResponse.json()
    print(urlResponse_json)
    # Retrieve the name from the url parameter /getmsg/?name=
    name = request.args.get("name", None)

    # For debugging
    print(f"Received: {name}")



    # ask the user for the location of the database
    #chatdb_location = input("Enter the absolute path of the chat database: ")
    #chatdb_location = "/Users/iliasboujlil/Library/Messages/chat.db"
    # ask the user for the location of the address book database:
    #address_book_location = input("Enter the absolute path of the address book database : ")
    #address_book_location = "/Users/iliasboujlil/Library/Application Support/AddressBook/Sources/985CDA7D-65ED-49D1-9A06-ECE5715B82AF/AddressBook-v22.abcddb"
    # ask the user for the number of messages to read
    n = 150
    #recent_messages = read_messages(chatdb_location, n)
    #print_messages(recent_messages)

    #addressBookData = get_address_book(address_book_location)
    #print(addressBookData)
    #combined_data = combine_data(recent_messages, addressBookData)
    #print_messages(combined_data)
    #messages=[{"role": "user", "content": "As an intelligent AI model, if you could be any fictional character, who would you choose and why?"}]
    message=[{"role": "user", "content": 'I need you to summarize the following input. it is a json. You must give a story type summary. Focus on the first_name and body fields. The first_name is the sender of the message and the body is the content of the message. Also only focus on the items with group_chat_name that includes Scoot FC. Create a story type narrative that is not too long based off of this text and make it 4 lines or less: {}'.format(urlResponse_json)}]
    #response = openai.Completion.create(
    #engine="text-davinci-003",
    #prompt="Summarize the text here in such a way that it is like a narrative and focus on the fields body and first_name. The field group_chat_name must also be Scoot FC:  {}".format(combined_data),
    #max_tokens=500,
    #)

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=message,
      temperature=0,
      max_tokens=100
    )
    #print(response)

    #response = {}

    # Check if the user sent a name at all
    
    # Return the response in json format
    url = "https://api.sendblue.co/api/send-message"

    headers = {
        "sb-api-key-id": SENDBLUE_API_KEY,
        "sb-api-secret-key": SENDBLUE_API_SECRET,
        "Content-Type": "application/json",
    }

    data = {
        "number": "+15133765542",
        "content": str(response.choices[0].message.content.strip()),
        "send_style": "",
        "media_url": "",
        "status_callback": "https://03ee-2603-7000-9200-966a-25dd-cda-2667-33e2.ngrok-free.app/msg"
    }

    response_send_blue = requests.post(url, json=data, headers=headers)
    if response_send_blue.status_code == 200:
        print(response_send_blue.json())
    else:
        print(f"Error: {response_send_blue.text}")
    print(response_send_blue)

    #response = sendblue.send_group_message(['+15133765542', '+15169967345'], response.choices[0].message.content.strip(), send_style='invisible')
    return jsonify(response.choices[0].message.content.strip())

@app.route('/post/', methods=['POST'])
def post_something():
    param = request.form.get('name')
    print(param)
    # You can add the test cases you made in the previous function, but in our case here you are just testing the POST functionality
    if param:
        return jsonify({
            "Message": f"Welcome {name} to our awesome API!",
            # Add this option to distinct the POST request
            "METHOD": "POST"
        })
    else:
        return jsonify({
            "ERROR": "No name found. Please send a name."
        })


@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Welcome to our medium-greeting-api!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)