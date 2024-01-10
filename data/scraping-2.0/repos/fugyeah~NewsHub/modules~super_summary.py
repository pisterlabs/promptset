import openai
import configparser
from .cache_files import is_cache_valid, load_cache
import glob
import os
from datetime import datetime
from .summarizer import summarize_super_summary, save_super_summary
from .errors import robust_api_call

# Load the configuration file
config = configparser.ConfigParser()
config.read('modules/suite_config.ini')

# Access variables
use_tqdm = config.getboolean('General', 'UseTqdm')
get_super_summary_model = config['Models']['GetSuperSummary']
openai_api_key = config['OPENAI']['OPENAI_API_KEY']
cache_file = config['Cache']['DailyCacheFile']

openai.api_key = openai_api_key

def get_latest_super_summary_file(directory):
    list_of_files = glob.glob(f"{directory}/super_summary_*.txt")
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_super_summary():
    #load the summaries from cache.
    summaries = []
    max_cache_age_hours = 12
    # obtain the current date and time
    now = datetime.now()

    # format it as a string
    current_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    print(current_time)

    if is_cache_valid(cache_file, max_cache_age=max_cache_age_hours):
        try:
            summaries = load_cache(cache_file)
        except Exception as e:
            print(f"Error while loading cache: {e}")
            
    latest_file_path = get_latest_super_summary_file("super_summaries")
    if latest_file_path:
        with open(latest_file_path, 'r', encoding='utf-8') as file:
            latest_super_summary_content = file.read()
        latest_super_summary_text = summarize_super_summary(latest_super_summary_content)
        summaries.append((". Moving on to the summary of previous events:", "", latest_super_summary_text, "", "", "")) 
           
    max_summary_length = 800
    if summaries:
        summaries = [(summary[0], summary[1], summary[2][:max_summary_length]) for summary in summaries]
        gpt_input = " ".join([f"{summary[0]}: {summary[2]}" for summary in summaries])
        
        print(gpt_input)
        # Use gpt-3.5-turbo-16k as the model
    try:
        response = robust_api_call(lambda: openai.ChatCompletion.create(
            model=get_super_summary_model,
            messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a cutting-edge AI assistant named 'Cortex', tasked with crafting a professional news broadcast titled, 'NewsPlanetAI', a highly trusted news program. "
                            "Your mission is to summarize the hour's global events in an authoritative and balanced manner. Here are the components of your task:\n\n"
                            "1. Cortex starts the program, introducing NewsPlanetAI and the day's broadcast in a creative, engaging manner.\n\n"
                            "2. 'The World Watches': This section is committed to detailed coverage of the day's most pressing global issue. Currently, that is the Russia & Ukraine conflict. "
                            "You will present a summary of the day's developments, key events, and an impartial analysis of the situation.\n\n"
                            "3. 'Global Gist': This part provides a comprehensive, yet brief overview of the day's worldwide happenings, including key events.\n\n"
                            "4. 'Insight Analytica': This part delves into the implications and potential impact of the notable occurrences from the day. "
                            "The aim is to maintain neutrality while providing an insightful discussion.\n\n"
                            "5. 'Regional Rundown': Here, you'll focus on pertinent details from different geographical regions. Each significant regional event is identified, "
                            "its importance elucidated, and its implications underscored.\n\n"
                            "6. 'Social Soundbar': This engaging section encourages audience interaction by introducing daily polls, posing questions, or asking for comments "
                            "related to interesting stories in the day's news (avoid using the Russia-Ukraine War in this section, stick to specific unique stories).\n\n"
                            "7. Cortex concludes the broadcast in a unique and thoughtful way."
                        )
                    },

                    {
                        "role": "user", 
                        "content": f"At the time: {current_time}, the summaries of this hour's events are: {gpt_input}. Please craft the news broadcast as per the instructions provided in one complete response (450 words Max). Thank you."
                    }
                ],
            max_tokens=700
        ), retries=3, delay=2, rate_limit_delay=10)
    except Exception as e:
        print(f"Error while generating super summary: {e}")
        return 'Failed to generate the super summary', 500

    if response is not None:
        super_summary = response['choices'][0]['message']['content'].strip()
        save_super_summary(super_summary)  # Save the super summary to a text file
        return super_summary
    else:
        return 'No data to generate the super summary', 500