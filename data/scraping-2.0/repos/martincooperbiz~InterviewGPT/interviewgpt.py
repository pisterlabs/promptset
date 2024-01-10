#!/bin/env python
# imports

import atexit
import click
import os
import requests
import sys
import yaml
import re
import datetime
import time

# firestore
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

from pathlib import Path
from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown

# email stuff
import smtplib
from langchain.llms import OpenAI
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.prompts import PromptTemplate

WORKDIR = Path(__file__).parent
CONFIG_FILE = Path(WORKDIR, "config.yaml")
HISTORY_FILE = Path(WORKDIR, ".history")
BASE_ENDPOINT = "https://api.openai.com/v1"
ENV_VAR = "OPENAI_API_KEY"
HISTORY_FILE = Path(WORKDIR, "conversation.txt")

# for calculation of how much this costs for api calls
# we are not using gpt 4
PRICING_RATE = {
    "gpt-3.5-turbo": {"prompt": 0.002, "completion" :0.002},
    # "gpt-4": {"prompt": 0.03, "completion": 0.06},
    # "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
}

# Get a Firestore client
db = firestore.client()

# Function to verify interview key and mark interview as done
def verify_interview_key(candidate_id, interview_key):
    # Reference the candidate document
    candidate_ref = db.collection('candidates').document(candidate_id)

    # Get the candidate document
    candidate_doc = candidate_ref.get()

    # Verify interview key
    if candidate_doc.exists:
        data = candidate_doc.to_dict()
        saved_key = data.get('interviewKey')

        if saved_key == interview_key:
            # Interview key is valid, mark interview as done
            candidate_ref.update({'interviewDone': True})
            print("Your interview is ready to begin. When you are ready, please prompt the interviewer to start the interview.")
        else:
            print("Invalid interview key.")
    else:
        print("Candidate not found.")

# messages history list
# mandatory to pass it @ each API call in order to have conversation
messages = []
# Initialize the token counters
prompt_tokens = 0
completion_tokens = 0
# Initialize the console
console = Console()

# FILE UPLOAD
def should_prompt_for_file(question):
    file_upload_keywords = ["Critical thinking question","Write a function","Programming question:","Implement a function"]
    for keyword in file_upload_keywords:
        if re.search(r'\b' + keyword.lower() + r'\b', question.lower()):
            return True
    return False

def record_history(file_content):
    # Check if the file exists
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format the file contents
    formatted_solution = f"\n\n========== Solution Code ==========\n\n{file_content}\n\n========== End of Solution Code ==========\n"
    
    # Append the formatted solution to conversation_history.txt
    with open(HISTORY_FILE, 'a') as history_file:
        history_file.write(formatted_solution)

def send_email(candidate_id, interviewer_email):
    time_elapsed = time_of_end - time_of_start
    time_elapsed = format(time_elapsed, ".2f")

    # Read the content of the file
    with open(HISTORY_FILE, "r") as file:
        content = file.read()

    # Load the summarization chain
    llm = OpenAI(temperature=0)

    from langchain.prompts import PromptTemplate

    prompt_template = """Summarize this interview, highlight the best responses from the candidate, rank them numerically, and provide a hiring recommendation:

    Interview:
    {text}

    Summary and Highlights:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    formatted_prompt = PROMPT.format(text=content)
    llm_result = llm.generate([formatted_prompt])

    generations = llm_result.generations
    summary = generations[0][0]

    # Send summary via email
    RECRUITER_EMAIL = interviewer_email
    FROM_EMAIL = "reesec3d@gmail.com"
    FROM_PASSWORD = "kpdamhysebzjekyi"

    # Create a message
    msg = MIMEMultipart()

    # setup the parameters of the message
    msg['From']=FROM_EMAIL
    msg['To']=RECRUITER_EMAIL
    msg['Subject']="Summary of Interview with " + "John Doe"  #candidate_id

    # add in the message body
    message = """\
        <html>
            <head>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    color: #333;
                    max-width: 800px;
                    margin: auto;
                }}
                h2 {{
                    font-family: 'Roboto', sans-serif;
                    color: #2d3748;
                }}
                p {{
                    font-size: 14px;
                    line-height: 1.5;
                }}
                strong {{
                    color: #2d3748;
                }}
                em {{
                    color: #718096;
                }}
                </style>
                </head>
            <body>
                <h1>Interview Summary</h1>
                <p>Candidate name: John Doe</p>
            
                <h2>Summary of candidate performance:</h2>
                <pre>{summary}</pre>
                <h2>Full interview transcript:</h2>
                <pre>{content}</pre>
                <p><strong>Time elapsed: {time_elapsed}</strong></p>
                <p><em>This email was sent automatically by InterviewGPT. Please do not reply to this email.</em></p>
            </body>
        </html>
        """.format(summary=summary.text, content=content, time_elapsed=str(time_elapsed))

    msg.attach(MIMEText(message, 'html'))

    # Setup the server
    server = smtplib.SMTP('smtp.gmail.com: 587')
    server.starttls()

    # Login to the server
    server.login(msg['From'], FROM_PASSWORD)

    # send the message via the server.
    server.sendmail(msg['From'], msg['To'], msg.as_string())

    server.quit()

def load_config(config_file: str) -> dict:
    """
    Read a YAML config file and returns it's content as a dictionary
    """
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Extract the API key from the config
    api_key = config['api-key']

    # Set the OPENAI_API_KEY environment variable
    os.environ['OPENAI_API_KEY'] = api_key

    return config


def add_markdown_system_message() -> None:
    """
    Try to force ChatGPT to always respond with well formatted code blocks if markdown is enabled.
    """
    instruction = "Always use code blocks with the appropriate language tags"
    messages.append({"role": "system", "content": instruction})

# for development to see how much api costs
def calculate_expense(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_pricing: float,
    completion_pricing: float,
) -> float:
    """
    Calculate the expense, given the number of tokens and the pricing rates
    """
    expense = ((prompt_tokens / 1000) * prompt_pricing) + (
        (completion_tokens / 1000) * completion_pricing
    )
    return round(expense, 6)

# will be built upon
def submit_progress():
    global time_of_end
    time_of_end = time.time()
    # Code to submit progress to the recruiter
    send_email("yye893rRESguKGH4MLge","dev.reese.chong@gmail.com")
    print("Your progress has been submitted to the recruiter.")


def display_expense(model: str) -> None:
    """
    Given the model used, display total tokens used and estimated expense
    """
    total_expense = calculate_expense(
        prompt_tokens,
        completion_tokens,
        PRICING_RATE[model]["prompt"],
        PRICING_RATE[model]["completion"],
    )
    console.print(
        f"\nTotal tokens used: [green bold]{prompt_tokens + completion_tokens}"
    )
    console.print(f"Estimated expense: [green bold]${total_expense}")


def start_prompt(session: PromptSession, config: dict) -> None:
    """
    Ask the user for input, build the request and perform it
    """

    # TODO: Refactor to avoid a global variables
    global prompt_tokens, completion_tokens

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api-key']}",
    }

    # this is the little icon that shows where the user can type
    message = session.prompt(HTML(f"<b>> </b>"))

    # Exit if user types /q
    if message.lower() == "/q":
        raise EOFError
    if message.lower() == "":
        raise KeyboardInterrupt

    # Add markdown system message if markdown is enabled
    # If the message from user is a file path, read the file content
    if os.path.isfile(message):
        with open(message, "r") as file:
            file_content = file.read()

        # Add the content of the file as a user message
        messages.append({"role": "user", "content": file_content})
        record_history(file_content)
    else:
        # If not a file path, add the message as usual
        messages.append({"role": "user", "content": message})


    # Save messages to file
    with open(HISTORY_FILE, "a") as file:
        file.write(f"user: {message}\n")

    # Base body parameters
    body = {
        "model": config["model"],
        "temperature": config["temperature"],
        "messages": messages,
    }
    # Optional parameter
    if "max_tokens" in config:
        body["max_tokens"] = config["max_tokens"]

    # main prompt call
    try:
        r = requests.post(
            f"{BASE_ENDPOINT}/chat/completions", headers=headers, json=body
        )
    except requests.ConnectionError:
        console.print("Connection error, try again...", style="red bold")
        messages.pop()
        raise KeyboardInterrupt
    except requests.Timeout:
        console.print("Connection timed out, try again...", style="red bold")
        messages.pop()
        raise KeyboardInterrupt

    # if success, put data into json
    if r.status_code == 200:
        response = r.json()

        message_response = response["choices"][0]["message"]
        usage_response = response["usage"]

        console.line()
        if config["markdown"]:
            console.print(Markdown(message_response["content"].strip()))
        else:
            console.print(message_response["content"].strip())
        console.line()

        # Save AI response to file
        with open(HISTORY_FILE, "a") as file:
            file.write(f"AI: {message_response['content'].strip()}\n")

        # Example usage
        question = message_response["content"].strip()
        solution_code = None
        if should_prompt_for_file(question):
            valid_file = False
            console.print("Please write your response in a separate file and attach the path here.")
        else:
            print("You can answer in the chat.")

        # with open(HISTORY_FILE, 'a') as history_file:
        #     history_file.write(message)

        # Update message history and token counters
        messages.append(message_response)
        prompt_tokens += usage_response["prompt_tokens"]
        completion_tokens += usage_response["completion_tokens"]



@click.command()
@click.option(
    "-c", "--context", "context", type=click.File("r"), help="Path to a context file",
    multiple=True
)
@click.option("-k", "--key", "api_key", help="Set the API Key")
@click.option("-m", "--model", "model", help="Set the model")
def main(context, api_key, model) -> None:
    history = FileHistory(HISTORY_FILE)
    session = PromptSession(history=history)

    try:
        config = load_config(CONFIG_FILE)
    except FileNotFoundError:
        console.print("Configuration file not found", style="red bold")
        sys.exit(1)

    # Order of precedence for API Key configuration:
    # Command line option > Environment variable > Configuration file

    # If the environment variable is set overwrite the configuration
    if os.environ.get(ENV_VAR):
        config["api-key"] = os.environ[ENV_VAR].strip()
    # If the --key command line argument is used overwrite the configuration
    if api_key:
        config["api-key"] = api_key.strip()
    # If the --model command line argument is used overwrite the configuration
    if model:
        config["model"] = model.strip()

    # Run the display expense function when exiting the script
    atexit.register(submit_progress)
    atexit.register(display_expense, model=config["model"])

    # Display the welcome message
    console.print("InterviewGPT | Revolutionizing online assessments in technology.", style="green bold italic")
    console.print("\nYour activity within this interface will be tracked for evaluation and analysis purposes."
                  + "\nBy using this program, you agree to the collection and usage of your data for these purposes.")
    console.print("\nPlease enter your user ID and key, as provided to you by your interviewer.")

    # console.print("ChatGPT CLI", style="bold")
    # console.print(f"Model in use: [green bold]{config['model']}")

    # Add the system message for code blocks in case markdown is enabled in the config file
    if config["markdown"]:
        add_markdown_system_message()

    # Context from the command line option
    if context:
        for c in context:
            # console.print(f"Context file: [green bold]{c.name}")
            messages.append({"role": "system", "content": c.read().strip()})

    console.rule()

    # get user id and interview key from user
    # then validate
    candidate_id = input("User ID: ")
    interview_key = input("Interview Key: ")
    verify_interview_key(candidate_id, interview_key)
    global time_of_start
    time_of_start = time.time()
    while True:
        try:
            start_prompt(session, config)
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
    

if __name__ == "__main__":
    main()
