import os
from dotenv import load_dotenv
from openai import OpenAI
from get_insta_posts import return_captions
import api_key

def return_formated_events(profiles,session_file):
    api_key_value=None
    try: #first we try to get api key from api_key.py
        api_key_value = api_key.get_api_key() # Get the API key from api_key.py
        client = OpenAI(api_key=api_key_value)
        print("Got API key from api_key.py")
    except Exception as e:
        print(f"API key error: {e}")
        try: #if api key is not found in api_key.py, then try to get it from .env variables
            load_dotenv()
            api_key_value = str(os.getenv("OPENAI_API_KEY")) #casting to string is necessary because of the way dotenv works
            client = OpenAI(api_key=api_key_value)
            print("Got API key from .env variables")
        except Exception as e:
            print(f"API key error: {e}")
    if api_key_value is None:
        print("API key could not be found. Please make sure you have an .env file with the OPENAI_API_KEY variable or a file called api_key.py with a get_api_key() function that returns the API key.")
        return None

    # List of Instagram profiles to process

    # Fetch captions from the specified Instagram profiles
    try:
        captions = return_captions(profiles,session_file)
    except Exception as e:
        print(f"Error while fetching captions: {e}")
        return None

    # Create a stream to process the captions using the OpenAI API
    try:
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user", 
                "content": f'data: {str(captions)}. Your job is to identify and extract events from the data and return their data in the following format: [{{"date" - date of the event, format %d.%m.%Y}},{{"name" - name of the event}},{{"description" - short description of the event (maximum 6 sentences)}}]. Please translate both the event names and descriptions into English. If an event lacks a specific date, skip it.'
            }],
            
            stream=True,
        )
    except Exception as e:
        print(f"Error while creating stream: {e}")
        return None

    # Process and append the output from the stream to an array
    output_array = []
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            output_array.append(chunk.choices[0].delta.content)

    combined_output = ''.join(output_array)

    return combined_output

    
                
                
if __name__ == "__main__":
    import pathlib
    profiles = ["aether_ry", "lahoevents", "koeputkiappro", "aleksinappro", "lasolary", "lymo.ry", "lirory", "Moveolahti", "koe_opku", "linkkiry"]
    session_file_path = str(pathlib.Path(__file__).parent.resolve()) # Get the path of the script
    session_file_name = "\\session-bencebansaghi" # Name of the session file
    session_file = session_file_path + session_file_name # Full path of the session file
    result = return_formated_events(profiles,session_file)
    print(result)

