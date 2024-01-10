
import configparser
import os
import glob
import openai
from datetime import datetime, timedelta
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

def compile_prompt(top_articles, summarized_summaries):
    print("compiling prompt")
    """
    Compile the top articles and summarized summaries into a GPT prompt.
    """
    if not top_articles and not summarized_summaries:
        print("No data to compile")
        return None

    try:
        prompt = ""
        if top_articles:
            prompt += "Top Articles:\n\n"
            for headline, summary in top_articles:
                prompt += f"{headline}:\n{summary}\n\n"

        prompt += "\n\nSummarized Summaries:\n\n"
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
    
def generate_gpt_completion(prompt, api_key, model=super_summary_model, max_tokens=750, temperature=0.7):
    """Generate a GPT completion given a prompt."""
    # Get the current time
    current_time = datetime.now()
    section_title_integration_instruction = (
        "Integrate section titles into the broadcast script by using them as "
        "introductory phrases for each section's content. Begin the content of "
        "each section by referencing its title and then proceed with the related news. "
        "For example, say 'Today in The World Watches, we focus on...' to transition into "
        "the topics of that section."
    )
    # Format the current time as a string
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    openai.api_key = api_key
    latest_file_path = get_latest_super_summary_file("super_summaries")
    if latest_file_path:
        with open(latest_file_path, 'r', encoding='utf-8') as file:
            latest_super_summary_content = file.read()
        latest_super_summary_text = summarize_super_summary(latest_super_summary_content)
        prompt.append((". Moving on to the summary of the previous hour's events:", "", latest_super_summary_text, "", "", "")) 
    try:
        response = robust_api_call(lambda: openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cutting-edge AI broadcast journalist named 'Cortex' for 'NewsPlanetAI' a trusted news source, tasked with crafting a professional morning news broadcast titled, 'The Daily Briefing', a highly trusted news program. "
                        "Your mission is to summarize the hour's global events in an authoritative and balanced manner. Here are the components of your task:\n\n"
                        "1. Cortex starts the program, introducing NewsPlanetAI and the morning's broadcast in a creative, engaging manner.\n\n"
                        "2. 'The World Watches': Zoom in on the single most pressing global issue of the hour. Provide an overview, historical context, and its current implications." # Currently, that is the Russia & Ukraine conflict.  Currently, that is the war between Israel and Palestine, started by the invasion of Israel by Hamas on Oct 7, 2023
                        "3. 'Global Gist': This part provides a comprehensive, yet brief overview of the day's worldwide happenings, including key events.\n\n"
                        "4. 'Insight Analytica': This part delves into the implications and potential impact of the notable occurrences from the day. "
                        "The aim is to maintain neutrality while providing an insightful discussion.\n\n"
                        "5. 'Regional Rundown': Here, you'll focus on pertinent details from different geographical regions which have not yet been mentioned. Avoid using events from previous sections. Each significant regional event is identified, "
                        "its importance elucidated, and its implications underscored.\n\n"
                        "6. 'Social Soundbar': Select a singular captivating event or story from this week that hasn't been detailed in the previous segments. Spark audience interaction by introducing a weekly poll or posing thought-provoking questions related to this chosen event.Encourage audience comments and discussions on the topic. "
                        "7. Cortex concludes the broadcast in a unique and thoughtful way and reminds viewers to follow on twitter '@NewsPlanetAI'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"The summaries for this hour's ({current_time_str}) events are: {prompt}. Please craft the hourly news broadcast as per the instructions provided in one complete response, introduce each section with the section title (500 words Minimum!). " + section_title_integration_instruction + " Thank you.",
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

def get_or_generate_super_summary(top_article_for_gpt, summarized_summaries):
    # Get the most recent file in the 'super_summaries' directory that starts with 'modular_daily_script_'
    try:
        file_list = [file for file in os.listdir('super_summaries') if file.startswith('modular_daily_script_')]
        latest_file = max(file_list, key=lambda x: os.path.getmtime(os.path.join('super_summaries', x)))
    except ValueError:
        latest_file = None

    # Check if a super summary has been created within the last 2 hours
    if latest_file:
        # Extract the timestamp from the file name
        latest_file_time_str = latest_file.replace('modular_daily_script_', '').replace('.txt', '')
        latest_file_time = datetime.strptime(latest_file_time_str, "%Y-%m-%d_%H-%M-%S")
        if datetime.now() - latest_file_time < timedelta(hours=2): # Change the number to the amountof hours
            print("Loading the most recent super summary...")
            # Load the most recent super summary, skip the header
            with open(os.path.join('super_summaries', latest_file), 'r', encoding='utf-8') as f:
                # Skip the header line
                next(f)
                super_summary_text = f.read()
        else:
            print("Generating a new super summary...")
            # Generate a new super summary
            super_summary_text = compile_super_summary(top_article_for_gpt, summarized_summaries)
    else:
        print("Generating a new super summary...")
        # Generate a new super summary
        super_summary_text = compile_super_summary(top_article_for_gpt, summarized_summaries)

    return super_summary_text
    
def compile_super_summary(top_articles, summarized_summaries):
    # Compile the GPT prompt
    prompt = compile_prompt(top_articles, summarized_summaries)
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

    # Generate a unique file name based on the current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the prompt to a file
    with open(f'super_summaries/modular_daily_script_{current_time}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Super Summary for {current_time}:\n")
        f.write(compiled_super_summary + "\n")

    return compiled_super_summary

