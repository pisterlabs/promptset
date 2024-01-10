import openai
import json
import sys
from datetime import datetime, timedelta
import re


def get_data(filename):
    import json

    # Read the JSON data from the file
    with open(filename, "r") as file:
        data_dict = json.load(file)

    return data_dict

def save_data(data_dict, filename):
    # Save the updated JSON data back to the same file
    with open(filename, "w") as file:
        json.dump(data_dict, file, indent=2)

def add_prompt_messages(role, content, messages):
    json_message = {
        "role": role, 
        "content": content
    }
    messages.append(json_message)
    return messages

def get_chat_completion_messages(messages, model="gpt-3.5-turbo", temperature=0.0): 
    try:
        response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    except Exception as e:
        print(e)
        sys.exit()
    else:
        return response.choices[0].message["content"]

def create_system_prompt():
    system_prompt = f"""
    You are an assistant that schedules showings on properties for a real estate agent.
    """
    return system_prompt

def create_user_prompt(prompt, data):
    now = get_current_datetime()
    prompt = f"""
    You need to double check you final answer and thought process.

    The current date and time is {now}
    
    User request: {prompt}

    Unavailable times: {data}

    MUST - The user request must include: 
        time
        date
        duration

    Available times are between 9 AM and 4 PM.

    When considering the request please validate the available and unavailable times.

    If you have time available, respond with the time you have available, and ask if they what to schedule that time. 

    If you do not have time available, respond with "no time available" and you can suggest something else

    Please let me know your thought process and final answer:
        
    """
    return prompt

def create_user_prompt_break_down(prompt):
    prompt = f"""
    Can you break down the user_prompt into constituent parts. 
    Request Type, Address, Event, Date, Time, Duration, and any other information you can find.
    Return your findings in the JSON object, no commentary
        request_type:
        address:
        event:
        date:
        time:
        duration:
        other:

    user_prompt: {prompt}
    """
    return prompt

def append_showing(data, address, new_start, new_end):
    found=False
    for listing in data["data"]["listings"]:
        if listing["listing"]["address"].lower() == address.lower():
            listing["listing"]["showings"].append({"start": new_start, "end": new_end})
            found =  True
            break
    return found, data

def query_database_by_address(database, target_address):
    json_data = get_data(database)
    listings = json_data["data"]["listings"]
    for listing in listings:
        address = listing["listing"]["address"]
        if address.lower() == target_address.lower():
            return listing["listing"]
    return None

def query_database_by_file(json_file):
    json_data = get_data(json_file)
    return json_data

def format_end_time(data):
    time = parse_time(data['time'])
    duration = parse_duration(data['duration'])
    new_duration = duration - timedelta(minutes=1)

    # Convert time to datetime object with today's date to perform addition
    today = datetime.now()
    end_time = datetime.combine(today.date(), time) + new_duration

    # Format the result in 12-hour format
    return end_time.strftime('%I:%M %p')

def get_current_datetime():
    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time in the desired format
    formatted_datetime = current_datetime.strftime("%m-%d-%Y %I:%M %p")

    return formatted_datetime

def parse_duration(duration_str):
    # Regular expression to extract the numeric value from the duration string
    duration_value = int(re.findall(r'\d+', duration_str)[0])
    
    # Determine the unit of time based on the duration string
    if 'm' in duration_str.lower():
        return timedelta(minutes=duration_value)
    elif 'h' in duration_str.lower():
        return timedelta(hours=duration_value)
    elif 'd' in duration_str.lower():
        return timedelta(days=duration_value)
    else:
        raise ValueError("Unsupported duration format. Use 'm', 'h', or 'd' for minutes, hours, or days.")

def parse_time(time_string):
    formats = ['%I:%M %p', '%I:%M%p', '%I%p', '%I %p']  # the different formats to try
    for fmt in formats:
        try:
            new_time = datetime.strptime(time_string, fmt).time()
            return new_time
        except ValueError:
            pass
    raise ValueError('no valid time format found')

def parse_date(date_str):
    # Define the possible date formats
    formats = [
        "%B %d",        # July 27
        "%B %dst",      # July 1st
        "%B %dnd",      # July 2nd
        "%B %drd",      # July 3rd
        "%B %dth",      # July 28th
        "%m/%d",        # 7/27
        "%m/%d/%Y",      # 7/28/2023
        "%m-%d",        # 7/27
        "%m-%d-%Y",      # 7/28/2023
    ]

    # Try parsing the date using each format
    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime("%m-%d-2023")
        except ValueError:
            pass

    # If no format matches, raise an error or return None based on your requirements
    raise ValueError("Invalid date format: {}".format(date_str))

