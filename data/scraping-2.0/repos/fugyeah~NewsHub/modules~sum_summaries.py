
import configparser
import os
import glob
import openai
from datetime import datetime
from modules.summarizer import summarize_super_summary
from modules.errors import robust_api_call
config = configparser.ConfigParser()
config.read('modules/suite_config.ini')



super_summary_model = config['Models']['GetSuperSummary']


def get_latest_super_summary_file(directory):
    list_of_files = glob.glob(f"{directory}/modular_super_summary_*.txt")
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def compile_prompt(summarized_summaries):
    print("compiling prompt")
    """
    Compile the summarized summaries into a GPT prompt.
    """
    if not summarized_summaries:
        print("No data to compile")
        return None

    try:
        prompt = ""
        for headline, summary in summarized_summaries:
            prompt += f"{headline}:\n{summary}\n\n"
        return prompt
    except Exception as e:
        print(f"Error while compiling prompt: {e}")
        return None



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
    
def generate_gpt_completion(prompt, api_key, model=super_summary_model, max_tokens=700, temperature=1.0):
    """Generate a GPT completion given a prompt."""
    # Get the current time
    current_time = datetime.now()

    # Format the current time as a string
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    openai.api_key = api_key
    latest_file_path = get_latest_super_summary_file("super_summaries")
    if latest_file_path:
        with open(latest_file_path, 'r', encoding='utf-8') as file:
            latest_super_summary_content = file.read()
        latest_super_summary_text = summarize_super_summary(latest_super_summary_content)
        prompt.append((". Moving on to the summary of previous events:", "", latest_super_summary_text, "", "", "")) 
    try:
        response = robust_api_call(lambda: openai.ChatCompletion.create(
            model=model,
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
                    ),
                },
                {
                    "role": "user",
                    "content": f"The summaries for this hour's ({current_time_str}) events are: {prompt}. Please craft the hourly news broadcast as per the instructions provided in one complete response (500 words Max). Thank you.",
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        ))
        if response is not None:
            return response.choices[0].message["content"]
        else:
            print("Error: Failed to generate GPT completion")
            return None
    except Exception as e:
        print(f"Error while generating GPT completion: {e}")
        return None

    
def compile_super_summary(summarized_summaries):
    # Compile the GPT prompt
    prompt = compile_prompt(summarized_summaries)
    print("Compiled Prompt:")
    print(prompt)

    # Load the OpenAI API key
    api_key = load_openai_api_key()
    print("Loaded API Key:")
    print(api_key)

    # Generate the GPT completion
    compiled_super_summary = generate_gpt_completion(prompt, api_key)
    print("GPT Completion:")
    print(compiled_super_summary)

    # If compiled_super_summary is None, return None immediately
    if compiled_super_summary is None:
        print("Error: Failed to generate GPT completion")
        return None

    # Get today's date
    today = datetime.today().strftime('%Y-%m-%d')  # format the date as 'YYYY-MM-DD'

    # Save the prompt to a file
    with open(f'super_summaries/modular_daily_script_{today}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Super Summary for {today}:\n")
        f.write(compiled_super_summary + "\n")

    return compiled_super_summary
