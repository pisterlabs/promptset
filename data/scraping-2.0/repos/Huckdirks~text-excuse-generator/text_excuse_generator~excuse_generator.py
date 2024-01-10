# Imported Libraries
from dotenv import load_dotenv
import openai   # I would just import the necessary functions, but api_key is too generic of a name, so I'm gonna keep the OpenAI namespace
import twilio.rest
from phonenumbers import is_valid_number, parse

# Python Libraries
from os import getenv, system
from os.path import dirname, join, isfile
from sys import argv

# Constants
ENV_NAME = ".env"  # CHANGE THIS TO YOUR ENVIRONMENT NAME (.env file)
ENV_PATH = join(dirname(__file__), ENV_NAME)


# Set up .env file
def setup_env(TWILIO_ACCOUNT_SID: str, TWILIO_AUTH_TOKEN: str, TWILIO_PHONE_NUMBER: str, OPENAI_API_KEY: str) -> bool:
    # Check if .env file exists and not empty
    if isfile(ENV_PATH):
        print(f"Error: \'{ENV_NAME}\' file already set up!")
        return False
    
    if not is_valid_number(parse(TWILIO_PHONE_NUMBER)):
        print("Error: Invalid phone number!")
        return False

    # Set up the .env file
    LINES = ["# Twilio API\n", f"TWILIO_ACCOUNT_SID = {TWILIO_ACCOUNT_SID}\n", f"TWILIO_AUTH_TOKEN = {TWILIO_AUTH_TOKEN}\n", f"TWILIO_PHONE_NUMBER = {TWILIO_PHONE_NUMBER}\n", "\n", "# OpenAI API\n", f"OPENAI_API_KEY = {OPENAI_API_KEY}\n", "\n", "# Phone Numbers to text\n"]

    # Write to the .env file
    with open(ENV_PATH, "w") as file:
        file.writelines(LINES)

    print(f"Set up \'{ENV_NAME}\' file!")
    return True


# Save a phone number to the .env file
def add_recipient(RECIPIENT: str, PHONE_NUMBER: str) -> bool:
    if not is_valid_number(parse(PHONE_NUMBER)):
        print("Error: Invalid phone number!")
        return False
    
    # Replace spaces with underscores for formatting
    NEW_RECIPIENT = RECIPIENT.replace(" ", "_")
    lines = []
    
    # Add a new user to the .env file
    with open(f"{ENV_NAME}", "r") as file:
        lines = file.readlines()

    if f"{NEW_RECIPIENT.upper()}_PHONE_NUMBER" in lines:
        print(f"Error: Recipient \'{NEW_RECIPIENT}\' already exists in .env file! Manually edit the .env file if you want to change the phone number.")
        return False
    
    INDEX = lines.index("# Phone Numbers to text\n")
    lines.insert(INDEX + 1, f"{NEW_RECIPIENT.upper()}_PHONE_NUMBER = \"{PHONE_NUMBER}\"\n")

    with open("personal_info.env", "w") as file:
        file.writelines(lines)

    print(f"Added recipient \'{NEW_RECIPIENT}\' with phone number \'{PHONE_NUMBER}\' to .env file!")
    return True


# If the -s or --send flag is given, send the text
def send_twilio_text(TO_PHONE_NUMBER: str, MESSAGE: str) -> None:
    if not isfile(ENV_PATH):
        print(f"Error: \'{ENV_NAME}\' file not set up! Use the -e or --setup_env flag to set up the .env file")
        return
    
    # Load the .env file
    load_dotenv()

    print("Sending text...")
    # Twilio API
    twilio_client = twilio.rest.Client(getenv("TWILIO_ACCOUNT_SID"), getenv("TWILIO_AUTH_TOKEN"))  # Login to Twilio
    twilio_client.messages.create(
        to = TO_PHONE_NUMBER,
        from_ = getenv("TWILIO_PHONE_NUMBER"),
        body = MESSAGE
    )
    
    print("Text sent!")
    return


# Generate an excuse and text it to a recipient. If no parameters are given, either by being passed in or given via the Command Line, it will prompt the user for input
def generate_excuse(**kwargs) -> (str | None):
    if kwargs and ("user" in kwargs and "recipient" in kwargs and "problem" in kwargs and "excuse" in kwargs):  # If parameters are passed in
        USER = kwargs["user"]
        RECIPIENT = kwargs["recipient"]
        PROBLEM = kwargs["problem"]
        EXCUSE = kwargs["excuse"]
        if "send_text" in kwargs:
            SEND_TEXT = kwargs["send_text"]
        if not USER or not RECIPIENT or not PROBLEM or not EXCUSE:
            print("Error: All parameters must be given!\nUsage: generate_excuse(user = \"\", recipient = \"\", problem = \"\", excuse = \"\", send_text = True)")
            return
        
    elif len(argv) == 4 or len(argv) == 5 or len(argv) == 6:        # If command line arguments are given, use them
        if len(argv) == 4 and (argv[1].lower() == "-a" or argv[1].lower() == "--add"):  # If the -a or --add flag is given with correct parameters, and correct # of parameters are passed in
            add_recipient(argv[2], argv[3])
            return
        elif (len(argv) == 6 and (argv[1].lower() == "-e" or argv[1].lower() == "--setup_env")):   # If the -e or --setup_env flag is given with correct parameters, and correct # of parameters are passed in
            setup_env(argv[2], argv[3], argv[4], argv[5])
            return
            
        # Load command line arguments
        USER = argv[1]
        RECIPIENT = argv[2]
        PROBLEM = argv[3]
        EXCUSE = argv[4]
        if len(argv) == 6 and (argv[5].lower() == "-s" or argv[5].lower() == "--send"):
            SEND_TEXT = True
        elif len(argv) == 6:
            print("\nError: Invalid flag given! Use -s or --send to send the text")
            return

    elif len(argv) == 1 and not kwargs:    # If no arguments are given, ask for user input
        if not isfile(ENV_PATH):  # If the .env file is not set up, ask the user if they want to set it up
            set_up_env_question = input(f"Error: \'{ENV_NAME}\' file not set up!\nDo you want to set it up now? (y/n): ")
            if set_up_env_question.lower() == "y" or set_up_env_question.lower() == "yes":
                TWILIO_ACCOUNT_SID = input("Enter your Twilio Account SID: ")
                TWILIO_AUTH_TOKEN = input("Enter your Twilio Auth Token: ")
                TWILIO_PHONE_NUMBER = input("Enter your Twilio Phone Number: ")
                OPENAI_API_KEY = input("Enter your OpenAI API Key: ")
                if not setup_env(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, OPENAI_API_KEY):
                    return
            else:
                print("Error: Cannot generate an EXCUSE without setting up the .env file!")
                return
            
        USER = input("Enter who is sending the text: ")
        RECIPIENT = input("Enter who you want to text: ")
        PROBLEM = input("Enter the fake problem you are having: ")
        EXCUSE = input("Enter the excuse you want to use: ")

        send_text_question = input("Do you want to send the text? (y/n): ")
        if send_text_question.lower() == "y" or send_text_question.lower() == "yes":
            SEND_TEXT = True

    else:
        # Give info on how to use the program if the wrong # of parameters are given
        system("clear")
        print("\nExcuse Generator Help:")
        print("\nThis program generates an excuse for you to send to someone. It uses the OpenAI API to generate the excuse, and the Twilio API to send the text.")
        print("\nThe program can be called in one of two ways:")
        print("\t1. From the command line (arguments are optional)")
        print("\tUsage: python3 text_excuse_generator.py [SENDER] [RECIPIENT] [PROBLEM] [EXCUSE] [--send_flag]\n\te.g. python3 text_excuse_generator.py \"John Doe\" \"Jane Doe\" \"Gonna miss dinner\" \"I'm sick\" -s")
        print("\n\t2. As parameters in a function call")
        print("\tUsage: generate_excuse(user = \"USER\", recipient = \"RECIPIENT\", problem = \"INSERT PROBLEM HERE\", excuse = \"INSERT EXCUSE HERE\", send_text = BOOL)\n\te.g. generate_excuse(user = \"John Doe\", recipient = \"Jane Doe\", problem = \"Gonna miss dinner\", excuse = \"I'm sick\", send_text = True)")
        print("\nThis function takes 4 required parameters and 1 optional parameter:")
        print("\tsender: The person who is sending the text")
        print("\trecipient: The person you want to text (can be saved person or a phone number)")
        print("\tproblem: The \"problem\" you are having")
        print("\texcuse: The excuse you want to use")
        print("\t--send_flag: If you want to send the text, add -s or --send. If you don't want to send the text, omit this flag\n")
        print("Or just run the program with no arguments to be prompted for input")
        print("Put any parameters longer than a single word in quotes, e.g. \"I'm sick\"\n")
        print("\nYou can also add new recipients to the .env file in one of two ways:")
        print("\t1. As command line arguments")
        print("\tUsage: python3 text_excuse_generator.py [-a/--add] [RECIPIENT] [PHONE_NUMBER]\n\te.g. python3 text_excuse_generator.py -a \"John Doe\" \"+15555555555\"")
        print("\n\t2. As parameters in a function call")
        print("\tUsage: add_recipient(\"RECIPIENT\", \"PHONE_NUMBER\")\n\te.g. add_recipient(\"John Doe\", \"+15555555555\")")
        print("\nThis function takes 2 required parameters:")
        print("\trecipient: The person you want to text (can be saved person or a phone number)")
        print("\tphone_number: The phone number you want to text the recipient at")
        print("\nYou can also set up the .env file in one of two ways:")
        print("\t1. As command line arguments")
        print("\tUsage: python3 text_excuse_generator.py [-e/--setup_env] [TWILIO_ACCOUNT_SID] [TWILIO_AUTH_TOKEN] [TWILIO_PHONE_NUMBER] [OPENAI_API_KEY]\n\te.g. python3 text_excuse_generator.py -e \"AC1234567890abcdef1234567890abcdef\" \"1234567890abcdef1234567890abcdef\" \"+15555555555\" \"sk-1234567890abcdef1234567890abcdef\"")
        print("\n\t2. As parameters in a function call")
        print("\tUsage: setup_env(\"TWILIO_ACCOUNT_SID\", \"TWILIO_AUTH_TOKEN\", \"TWILIO_PHONE_NUMBER\", \"OPENAI_API_KEY\")\n\te.g. setup_env(\"AC1234567890abcdef1234567890abcdef\", \"1234567890abcdef1234567890abcdef\", \"+15555555555\", \"sk-1234567890abcdef1234567890abcdef\")")
        print("\nThe prompt sent to ChatGPT is: \"Write a text message to [RECIPIENT] explaining that you [PROBLEM] because [EXCUSE]. Also start the message by stating this is [USER], and end the message by telling the recipient to text my actual phone number back if you really need me.\"\n")
        return
        

    # If the .env file is not set up, or is empty, return   
    if not isfile(ENV_PATH):
        print(f"Error: \'{ENV_NAME}\' file not set up! Use the -e or --setup_env flag to set up the .env file")
        return
    
    # Load the .env file
    load_dotenv()

    to_phone_number = ""
    if (RECIPIENT[0] == '+' and RECIPIENT[1:].isnumeric()) and is_valid_number(parse(RECIPIENT)):   # Check if RECIPIENT is a phone number
        to_phone_number = RECIPIENT
    elif SEND_TEXT:
        RECIPIENT_FORMATTED = RECIPIENT.replace(" ", "_")
        to_phone_number = getenv(f"{RECIPIENT_FORMATTED.upper()}_PHONE_NUMBER")
        if to_phone_number == None: # If the recipient is not in the .env file
            print(f"\nError: No phone number found for recipient \'{RECIPIENT}\' in .env file!")
            if len(argv) != 1 or not len(kwargs) == 0:  # If not in user input mode, exit, else ask if they want to add the recipient
                return
            
            # Ask if they want to add the recipient
            ADD_RECIPIENT_QUESTION = input("Do you want to add this recipient to the .env file? (y/n): ")
            if ADD_RECIPIENT_QUESTION.lower() == "y" or ADD_RECIPIENT_QUESTION.lower() == "yes":
                to_phone_number = input("Enter the phone number of the recipient: ")
                if not add_recipient(RECIPIENT, to_phone_number):   # If the recipient could not be added
                    return
            else:
                return

    # Create the message (AI Time!)
    CHATGPT_CONTEXT = f"Write a text message to {RECIPIENT} explaining that you {PROBLEM} because {EXCUSE}. Also start the message by stating this is {USER}, and end the message by telling the recipient to text my actual phone number back if you really need me."
    print("\nCreating message...\n")

    # OpenAI API
    openai.api_key = getenv("OPENAI_API_KEY")
    AI_QUERY = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": CHATGPT_CONTEXT}]
    )

    AI_RESPONSE = AI_QUERY.choices[0].message.content
    print(f"Chat GPT's Response:\n{AI_RESPONSE}\n")

    if SEND_TEXT:   # If the -s or --send flag is given, send the text
        send_twilio_text(to_phone_number, AI_RESPONSE)

    return AI_RESPONSE


if __name__ == "__main__":
    generate_excuse()