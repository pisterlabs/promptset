import logging
from datetime import datetime, timedelta
from telethon.sync import TelegramClient
import pytz
import configparser
import os
import openai

# Initialize the ConfigParser
config = configparser.ConfigParser()

# Read the suite_config.ini file
config.read('modules/suite_config.ini')

# Set up logging
logging_level = config.get('logging', 'level', fallback='WARNING')
logging.basicConfig(format='[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s',
                    level=getattr(logging, logging_level))

# Telegram API credentials
api_id = config.get('telegram', 'api_id')
api_hash = config.get('telegram', 'api_hash')
phone_number = config.get('telegram', 'phone_number')
# Retrieve FTP values from the config file
ftp_host = config.get('FTP', 'Host')
ftp_user = config.get('FTP', 'User')
ftp_password = config.get('FTP', 'Password')
ftp_directory = config.get('FTP', 'Directory')

utc_now = datetime.now(pytz.utc)
time_24_hours_ago = utc_now - timedelta(days=1)
from tiktoken import encoding_for_model

def count_tokens(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    # Use tiktoken.encoding_for_model() to automatically load the correct encoding for a given model name.
    encoding = encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_openai_api_key(config_file='modules/suite_config.ini'):
    """
    Load the OpenAI API key from a configuration file.
    """
    if not os.path.exists(config_file):
        print(f"No configuration file found at {config_file}")
        return None

    config = configparser.ConfigParser()
    config.read(config_file)

    try:
        api_key = config.get('OPENAI', 'OPENAI_API_KEY')
        return api_key
    except Exception as e:
        print(f"Error while loading OpenAI API key: {e}")
        return None
    
api_key = load_openai_api_key(config_file='modules/suite_config.ini')
if not api_key:
    print("Failed to load the OpenAI API key.")
    exit()
openai.api_key = api_key

    
def generate_gpt_completion(prompt, api_key, model='gpt-3.5-turbo-16k', max_tokens=1000, temperature=0.5):
    """Generate a GPT completion given a prompt focused on the war in Ukraine."""
    current_time = datetime.now()
    openai.api_key = api_key
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    root_directory = os.path.dirname(os.path.abspath(__file__))
    reference_file_path = os.path.join(root_directory, '..', 'russo_ukranian_war_abridged.txt')

    with open(reference_file_path, 'r', encoding='utf-8') as file:
        reference_text = file.read()
        print("Appending material on the Russo-Ukrainian War.")
        
    print("Sending telegram messages to GPT for processing:")

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are 'Cortex', an AI designed for advanced content aggregation and summarization. Your task is to process a list of current telegram messages related to the Russian-Ukraine conflict, consolidate similar ones."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Here are today's telegram messages dated ({current_time_str}) related to the Russia & Ukraine conflict: {prompt}. Please provide a structured summary, focusing on the most pressing information and consolidating repetitive messages. Thank you.",
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response is not None:
            print("response returned from GPT")
            return response.choices[0].message["content"], prompt, reference_text

        else:
            print("Error: Failed to generate GPT completion")
            return None, prompt, reference_text
    except Exception as e:
        print(f"Error while generating GPT completion: {e}")
        return None
    
def process_gpt_response(gpt_response, api_key, model='gpt-4-1106-preview', max_tokens=1000, temperature=0.5):
    """Generate a GPT completion given a prompt focused on the war in Ukraine."""
    current_time = datetime.now()
    openai.api_key = api_key
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    root_directory = os.path.dirname(os.path.abspath(__file__))
    reference_file_path = os.path.join(root_directory, '..', 'russo_ukranian_war_abridged.txt')

    with open(reference_file_path, 'r', encoding='utf-8') as file:
        reference_text = file.read()
        print("Appending material on the Russo-Ukrainian War.")
        
    # Prepare the prompt with historical reference
    context = f"\n\nHistorical reference material on the Russo-Ukrainian War:\n{reference_text}"
    print("Compiled context:")
    print(context)
    print("Sending the generated GPT summary back to GPT for analysis:")

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are 'Cortex', an AI designed for OSINT. Your task is to process a list of telegram messages related to the Russian-Ukraine conflict, consolidate similar ones, and create a cohesive and structured briefing. Ensure the summary is focused, informative, and avoids redundancies. Do not include or summarize the historical context in your summary; it's for reference purposes only."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Here are today's telegram messages dated ({current_time_str}) related to the Russia & Ukraine conflict: {gpt_response}. Please provide a structured summary, focusing on the most pressing information and consolidating repetitive messages. Consider this historical background as reference, thank you: {context}",
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response is not None:
            print("response returned from GPT")
            return response.choices[0].message["content"]

        else:
            print("Error: Failed to generate GPT completion")
            return None
    except Exception as e:
        print(f"Error while generating GPT completion: {e}")
        return None

from ftplib import FTP
def upload_to_ftp(file_path, host, user, password, remote_path, remote_filename):
    ftp = FTP(host)
    ftp.login(user, password)
    ftp.cwd(remote_path)
    with open(file_path, 'rb') as file:
        ftp.storbinary(f'STOR {remote_filename}', file)
    ftp.quit()
    
# Connect to Telegram
with TelegramClient('session_name', api_id, api_hash) as client:
    print("Connected to Telegram")
    
    if not client.is_user_authorized():
        print("Authorizing...")
        client.start(phone=phone_number)
        print("Authorized successfully")

    # List of channel usernames or IDs
    channel_list = ['@Pravda_Gerashchenko', '@pilotblog', '@aeronavtyua', '@rybar', '@annamaliar', '@operativnoZSU', '@dva_majors', '@mod_russia', '@prigozhin_2023_tg', '@RVvoenkor']  
  
    # Function to scrape recent messages from a channel
    #Limit reduced to 3 on high news days, default is 10
    def scrape_channel_messages(channel, limit=4):
        print(f"Scraping messages from {channel}...")
        try:
            # Fetch messages from the last 24 hours
            messages = [message for message in client.iter_messages(channel, limit=limit) if message.date > time_24_hours_ago]
            print(f"Fetched {len(messages)} messages from the last 24 hours from {channel}")
            return messages
        except Exception as e:
            logging.error(f"Error while scraping {channel}: {e}")
            return []


    # Initialize a list to hold the top 5 messages from each channel
    top_messages_from_each_channel = []

    # Loop through each channel in channel_list
    for channel in channel_list:
        # Fetch messages from the current channel
        messages = scrape_channel_messages(channel)
        
        # Sort the messages by views, then by number of replies (comments)
        sorted_messages = sorted(messages, key=lambda x: (x.views or 0, getattr(x.replies, 'replies', 0) if x.replies else 0), reverse=True)
        
        # Take the top 5 messages for the current channel
        top_5_messages = sorted_messages[:5]
        
        # Append these top 5 messages to the global list
        top_messages_from_each_channel.append({"channel": channel, "messages": top_5_messages})


    # Now, top_messages_from_each_channel contains the top 5 messages from each channel.
    messages_for_prompt = '\n'.join([f"{message.text}" for item in top_messages_from_each_channel for message in item['messages']])
    
    gpt_summary, returned_prompt, returned_reference_text = generate_gpt_completion(messages_for_prompt, api_key)

    prompt_tokens = count_tokens(returned_prompt)
    reference_text_tokens = count_tokens(returned_reference_text)
    response_tokens = count_tokens(gpt_summary)

    print(f"Tokens in prompt: {prompt_tokens}")
    print(f"Tokens in reference text: {reference_text_tokens}")
    print(f"Tokens in response: {response_tokens}")

    processed_result = process_gpt_response(gpt_summary, api_key)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    now = datetime.now()
    date_string = now.strftime("%B %d, %Y")
    file_path_with_timestamp = f'ukraine/telegram_summary_{timestamp}.txt'
    processed_result_timestamp = f'ukraine/telegram_summary_processed_{timestamp}.txt'
    with open(file_path_with_timestamp, 'w', encoding='utf-8') as f:
        f.write(gpt_summary)
    print(f"Summarized messages saved to: {file_path_with_timestamp}")

    with open(processed_result_timestamp, 'w', encoding='utf-8') as f:
        f.write(processed_result)
    print(f"Processed Response saved to: {processed_result_timestamp}")
# Step 1: Concatenate the two text files
    with open(file_path_with_timestamp, 'r', encoding='utf-8') as f:
        gpt_summary_content = f.read()

    with open(processed_result_timestamp, 'r', encoding='utf-8') as f:
        processed_result_content = f.read()

    # combined_content = processed_result_content + "\n\n" + gpt_summary_content

    # Step 2: Create an HTML file from the HTML string
    title = "Ukraine OSINT Summary"  # Set the title for your HTML page
    sources_string = "<ul>" + "".join([f"<li>{channel}</li>" for channel in channel_list]) + "</ul>"
    briefing_text = processed_result_content.replace("\n", "<br/>")  # Convert newline to <br> for HTML
    processed_telegram = gpt_summary_content.replace("\n", "<br/>")  # Convert newline to <br> for HTML
    html_string = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8"/>
        <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
        <title>{title}</title>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css">
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                background: #f5f5f5;
                color: #333;
                line-height: 1.6;
            }}
            .navbar {{
                background: #343a40;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }}
            .container {{
                background-color: #fff;
                width: 80%;
                color: #333;
                padding: 20px;
                border-radius: 10px;
                margin: 20px auto;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }}
            h1 {{
                font-size: 2.2rem;
                margin-bottom: 20px;
                border-bottom: 2px solid #f0f0f0;
                padding-bottom: 10px;
            }}
            h2 {{
                margin-bottom: 20px;
                border-bottom: 2px solid #f0f0f0;
                padding-bottom: 10px;
            }}            
            .summary p {{
                font-size: 1.1rem;
                line-height: 1.7;
            }}
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <a class="navbar-brand" href="/">NewsPlanetAi</a>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/ukraine">Ukraine Daily Report</a>
                    </li>
                </ul>
            </div>
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
        </button>
        </nav>
        <div class="container">
            <h1>Ukraine OSINT Report</h1>
            <div class="summary">
                <h2>OSINT Briefing for {date_string}:</h2>
                <p>{briefing_text}</p>
                <hr>
                <h2>Telegram Messages Processed:</h2>
                <p>{processed_telegram}</p>
                <h2>Sources:</h2>
                <p>{sources_string}</p>
            </div>

            </div>
        </div>
    </body>
    </html>
    """
    html_file_name = f'ukraine/telegram_html_summary_{timestamp}.html'
    with open(html_file_name, 'w', encoding='utf-8') as html_file:
        html_file.write(html_string)

    print(f"HTML report saved to: {html_file_name}")
    # Function to upload a file to an FTP server
    
    # Step 3: Upload the HTML file to your server using the provided variables
    remote_filename = 'osint_report.html'
    upload_to_ftp(html_file_name, ftp_host, ftp_user, ftp_password, ftp_directory + 'ukraine/', remote_filename)

    print(f"HTML report uploaded to server as: {remote_filename}")
