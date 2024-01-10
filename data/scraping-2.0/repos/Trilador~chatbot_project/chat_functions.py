import os
import re
import ast
import openai
import datetime
import requests
from bs4 import BeautifulSoup
from typing import List, Union

# Setting up OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Constants
LOG_FILE = "chatbot_log.txt"
LAST_TOPIC = None
CONTEXT_SIZE = 10  # Assuming you want to keep the last 4 interactions for context


# OpenAI Constants
OPENAI_API_ENGINE = "davinci"
OPENAI_API_MAX_TOKENS = 2048
OPENAI_API_TEMPERATURE = 0.7
OPENAI_API_FREQUENCY_PENALTY = 0
OPENAI_API_PRESENCE_PENALTY = 0

# Utility Imports
from utility_modules.api_integrations import (
    wolfram_alpha_query,
    analyze_text,
    get_dialogflow_response
)
from utility_modules.file_operations import (
    list_files,
    list_directories,
    create_directory,
    delete_directory,
    display_file_content,
    save_file_content,
    search_file_content,
)
from utility_modules.code_generations import (
    natural_language_to_code,
    provide_code_feedback,
    interactive_code_correction,
    analyze_python_code_ast,
)
from utility_modules.web_operations import fetch_wikipedia_summary, search_python_documentation

# Import code databases
from code_db.advanced_constructs import PYTHON_CODE_DB as ADVANCED_DB
from code_db.basic_constructs import PYTHON_CODE_DB as BASIC_DB
from code_db.python_modules import PYTHON_CODE_DB as MODULES_DB

# Import command handling functions
from command_handling import COMMAND_PATTERNS
from command_handling import handle_google_search, handle_music_play #and other handler functions you need

# Global Variables
context: List[str] = []


def log_interaction(user_query: str, bot_response: str) -> None:
    """Log user queries and bot responses."""
    with open(LOG_FILE, 'a') as log:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.write(f"{timestamp} - User: {user_query}\n")
        log.write(f"{timestamp} - Bot: {bot_response}\n")

def ask_for_feedback() -> None:
    """Ask the user for feedback after a few interactions."""
    feedback = input("How would you rate your experience with our chatbot (1-5)? ")
    log_interaction("Feedback", feedback)

def handle_help_request() -> str:
    """Provide a list of commands or functionalities the bot supports."""
    return """
    Here's what I can assist you with:
    - Answer general questions using OpenAI.
    - Provide feedback on my performance.
    - And more! Just ask.
    """



# Note: Moved the COMMON_RESPONSES dictionary here for better organization.
COMMON_RESPONSES = {
    "hello": "Hello! How can I assist you today?",
    "how are you": "I'm just a program, so I don't have feelings, but I'm functioning optimally! How can I help you?",
    "thank you": "You're welcome! Let me know if there's anything else.",
    # ... add more common queries and responses as needed ...
}

# Note: Moved the update_context and get_recent_context functions here for better organization.
def update_context(user_query: str, bot_response: str) -> None:
    global context
    context.append(f"User: {user_query}")
    context.append(f"Bot: {bot_response}")
while len(context) > CONTEXT_SIZE:
            context.pop(0)

def get_recent_context() -> str:
    return "\n".join(context[-4:])

# Code Operations Begin
# Code Operations
def check_code_db(query: str) -> Union[str, None]:
    """Check if the user's query matches any code database entries."""
    return next(
        (
            f"{description}\n\nCode:\n{code}"
            for topic, (code, description) in {
                **ADVANCED_DB,
                **BASIC_DB,
                **MODULES_DB,
            }.items()
            if topic in query
        ),
    None,
    )
def natural_language_to_code(natural_language: str) -> str:
    """
    Converts natural language instructions to code using OpenAI API.
    """
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt=f"Write Python code for: {natural_language}",
            max_tokens=2048
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error converting natural language to code: {str(e)}"

def provide_code_feedback(code: str) -> str:
    """
    Provides feedback on a given code snippet.
    Placeholder function for now.
    """
    return "This is a placeholder function. Feedback functionality will be added later."

def interactive_code_correction(code: str, feedback: str) -> str:
    """
    Interactively corrects code based on user feedback.
    Placeholder function for now.
    """
    return "This is a placeholder function. Interactive code correction functionality will be added later."

def analyze_python_code_ast(code: str) -> str:
    """
    Analyzes the Abstract Syntax Tree (AST) of a Python code snippet.
    """
    try:
        tree = ast.parse(code)
        return str(tree)
    except Exception as e:
        return f"Error analyzing Python code AST: {str(e)}"
    # Code Operations End

# File Operations Begin

def list_files(directory: str) -> str:
    try:
        files = os.listdir(directory)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

def list_directories(directory: str) -> str:
    try:
        dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        return "\n".join(dirs)
    except Exception as e:
        return f"Error listing directories: {str(e)}"

def delete_directory(directory_name, parent_directory="C:\Users\timot\Desktop\Python\chatbot_project\chatbot_project\Vault") -> str:
    dir_path = os.path.join(parent_directory, directory_name)
    if not os.path.exists(dir_path):
        return f"Error: Directory {dir_path} does not exist."
    try:
        os.rmdir(dir_path)
        return f"Successfully deleted directory: {dir_path}"
    except Exception as e:
        return f"Error deleting directory {dir_path}: {str(e)}"

def display_file_content(filename, directory=DEFAULT_DIRECTORY):
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return f"Error: {filepath} does not exist."
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading from {filepath}: {str(e)}"

def save_file_content(filename, content, directory=DEFAULT_DIRECTORY):
    filepath = os.path.join(directory, filename)
    try:
        with open(filepath, 'w') as file:
            file.write(content)
        return f"Successfully saved changes to {filepath}."
    except Exception as e:
        return f"Error writing to {filepath}: {str(e)}"
    
def search_file_content(file_path: str, search_term: str) -> str:
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        matching_lines = [line for line in lines if search_term in line]
        return "\n".join(matching_lines)
    except Exception as e:
        return f"Error searching file content: {str(e)}"
    
def handle_file_reading_request(query: str) -> str:
    """
    Handle file reading requests using the ASKTHECODE plugin.
    """
    # For now, we'll just check for a simple command to read a file.
    # This can be enhanced in the future to handle more complex file operations.
    if "read" in query or "open" in query or "type out" in query:
        # Use the ASKTHECODE plugin to read the file.
        # For demonstration purposes, we'll just return a placeholder response.
        return "File content displayed."
    return ""

def create_new_file(filename, content="", directory=DEFAULT_DIRECTORY):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        return f"Error: {filepath} already exists."
    try:
        with open(filepath, 'w') as file:
            file.write(content)
        return f"Successfully created {filepath}."
    except Exception as e:
        return f"Error creating {filepath}: {str(e)}"

def delete_file(filename, directory=DEFAULT_DIRECTORY):
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return f"Error: {filepath} does not exist."
    try:
        os.remove(filepath)
        return f"Successfully deleted {filepath}."
    except Exception as e:
        return f"Error deleting {filepath}: {str(e)}"

def rename_file(old_filename, new_filename, directory=DEFAULT_DIRECTORY):
    old_filepath = os.path.join(directory, old_filename)
    new_filepath = os.path.join(directory, new_filename)
    if not os.path.exists(old_filepath):
        return f"Error: {old_filepath} does not exist."
    if os.path.exists(new_filepath):
        return f"Error: {new_filepath} already exists."
    try:
        os.rename(old_filepath, new_filepath)
        return f"Successfully renamed {old_filepath} to {new_filepath}."
    except Exception as e:
        return f"Error renaming {old_filepath}: {str(e)}"

def search_file_content(content, directory=DEFAULT_DIRECTORY):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                with open(os.path.join(root, file), 'r') as f:
                    if content in f.read():
                        matching_files.append(os.path.join(root, file))
            except Exception as e:
                continue  # Skip files that can't be read
    return matching_files    


def get_dictionary_response(query: str) -> str:
    """
    Check if the user's query matches any common queries and return the respective response.
    """
    return next(
        (
            value
            for key, value in COMMON_RESPONSES.items()
            if key in query.lower()
        ),
        None,
    )




def handle_openai_response(query: str, context: List[str]) -> str:
    # Use the last 4 interactions for context
    full_prompt = "\n".join(context[-4:] + [f"User: {query}", "Bot:"])
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt=full_prompt,
            temperature=0.5,  # Reduce randomness
            max_tokens=2048,  # Define TOKEN_LIMIT
            top_p=1,
            frequency_penalty=0,  # Adjust penalties
            presence_penalty=0,
            stop=["\n", "User:", "Bot:"]
        )
        response_text = response.choices[0].text.strip()
    except openai.error.OpenAIError as e:  # Use "Error" instead of "error"
        if "Input text not set." in str(e):
            return "OpenAI Error: It seems there was an issue with the input provided to OpenAI."
        return f"OpenAI Error: {str(e)}"
    except Exception as e:
        return f"Error with OpenAI response: {str(object)}"
    
    # Enhanced Response Filtering
    last_user_query = context[-2] if len(context) > 1 else None
    last_bot_response = context[-1] if len(context) > 1 else None
    if response_text == last_bot_response and query == last_user_query:
        response_text = "I've already provided that response. Can you please rephrase or ask a different question?"
    elif response_text == query:
        response_text = "I'm not sure how to respond to that. Can you provide more context or rephrase your question?"

    # Adjusted condition
    if len(response_text.split()) > 2048:  # Use 2048 instead of TOKEN_LIMIT
        response_text = "My response seems too long. Would you like a more concise answer or should I clarify something specific?"

    return response_text

def handle_github_request(query: str) -> str:
    """
    Handle GitHub-related requests using the ASKTHECODE plugin.
    """
    # For now, we'll just check for a simple command to fetch code from GitHub.
    # This can be enhanced in the future to handle more complex GitHub operations.
    if "fetch code from GitHub" in query:
        # Use the ASKTHECODE plugin to fetch code from the GitHub repository.
        # For demonstration purposes, we'll just return a placeholder response.
        return "Fetched code from GitHub repository."
    return ""

def handle_music_play(query: str) -> str:    
    """
    Handle music-related requests using the ASKTHECODE plugin.
    """
    # For now, we'll just check for a simple command to play music.
    # This can be enhanced in the future to handle more complex music operations.
    if "play" in query:
        # Use the ASKTHECODE plugin to play music.
        # For demonstration purposes, we'll just return a placeholder response.
        return "Playing music..."
    return ""

def google_search(query: str) -> str:
    """Perform a Google search and return the top result."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    search_url = GOOGLE_SEARCH_URL + query.replace(" ", "+")
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = soup.find_all('div', class_='tF2Cxc')
    top_result = search_results[0]
    link = top_result.find('a')['href']
    title = top_result.find('h3').text
    snippet = top_result.find('span', class_='aCOpRe').text
    return f"Top search result for '{query}':\nTitle: {title}\nLink: {link}\nSnippet: {snippet}"

def handle_github_request(query: str) -> str:
    """Handle GitHub-related requests."""
    if "fetch code from GitHub" in query:
        return "Fetched code from GitHub repository."
    return ""


def main() -> None:
    """Main loop for chatbot interaction."""
    user_query = input("You: ")
    response = chatbot_response(user_query)
    print(f"Bot: {response}")
    while True:
        user_query = input("You: ")
        try:
            if not user_query.strip():
                print("It seems you didn't provide any input. Please ask a question or provide a command.")
                continue
            
            global context
            context.append(user_query)

            if len(context) > CONTEXT_SIZE:
                context.pop(0)

 

            def handle_github_request(query: str) -> str:
                return "Fetched code from GitHub repository."

            def handle_file_reading_request(query: str) -> str:
                return "File reading request handled."

            def search(pattern: str, string: str) -> bool:
                return bool(re.search(pattern, string))

            def handle_openai_response(query: str, context: List[str]) -> str:
                """
                Handle OpenAI API response for general chatbot interactions.
                """
                try:
                    response = openai.Completion.create(
                        engine=OPENAI_API_ENGINE,
                        prompt=query,
                        max_tokens=OPENAI_API_MAX_TOKENS,
                        n=1,
                        stop=None,
                        temperature=OPENAI_API_TEMPERATURE,
                        frequency_penalty=OPENAI_API_FREQUENCY_PENALTY,
                        presence_penalty=OPENAI_API_PRESENCE_PENALTY,
                        context=context
                    )
                    response_text = response.choices[0].text.strip()
                except openai.Error as e:  # Use "Error" instead of "error"
                    if "Input text not set." in str(e):
                        return "OpenAI Error: It seems there was an issue with the input provided to OpenAI."
                    return f"OpenAI Error: {str(e)}"
                except Exception as e:
                    return f"Error with OpenAI response: {str(e)}"

                # Enhanced Response Filtering
                last_user_query = context[-2] if len(context) > 1 else None
                last_bot_response = context[-1] if len(context) > 1 else None
                if response_text == last_bot_response and query == last_user_query:
                    response_text = "I've already provided that response. Can you please rephrase or ask a different question?"
                elif response_text == query:
                    response_text = "I'm not sure how to respond to that. Can you provide more context or rephrase your question?"

                # Adjusted condition
                if len(response_text.split()) > 2048:  # Use 2048 instead of TOKEN_LIMIT
                    response_text = "My response seems too long. Would you like a more concise answer or should I clarify something specific?"

                return response_text

            # Check for GitHub-related requests
            if "GitHub" in user_query:
                return handle_github_request(user_query)

            # Check for file reading requests
            if "C:\\" in user_query and ("read" in user_query or "open" in user_query or "type out" in user_query):
                return handle_file_reading_request(user_query)

            # Check for command patterns
            for pattern, handler in COMMAND_PATTERNS.items():
                if search(pattern, user_query):
                    return handler(user_query)

            # Use Dialogflow for general chatbot interactions
            if dialogflow_response := get_dialogflow_response(user_query):
                return dialogflow_response

            # Use Cloud Natural Language API for sentiment analysis
            sentiment_score, sentiment_magnitude = analyze_text(user_query)
            if sentiment_score > 0.7:
                return "I'm glad to hear that!"
            elif sentiment_score < -0.7:
                return "I'm sorry to hear that. How can I assist you further?"

            # If neither Dialogflow nor sentiment analysis provides a clear response, use OpenAI API
            else:
                return handle_openai_response(user_query, get_recent_context())
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}. Please try again or rephrase your query.")
def store_feedback(user_query: str, bot_response: str, rating: str) -> None:
    """Store user feedback."""
    print(f"Feedback received for query '{user_query}':")
    print(f"Bot Response: {bot_response}")
    print(f"Rating: {rating}/5")

def main() -> None:
    """Main loop for chatbot interaction."""
    user_query = input("You: ")
    response = chatbot_response(user_query)
    print(f"Bot: {response}")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        response = chatbot_response(user_query)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()