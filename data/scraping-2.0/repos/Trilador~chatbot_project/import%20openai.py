test*.py
import openai
import os
from typing import List, Union
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
from code_db.advanced_constructs import PYTHON_CODE_DB as ADVANCED_DB
from code_db.basic_constructs import PYTHON_CODE_DB as BASIC_DB
from code_db.python_modules import PYTHON_CODE_DB as MODULES_DB

# Set up the OpenAI API
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

DEFAULT_DIRECTORY = "C:\\Users\\timot\\Desktop\\Python"
TOKEN_LIMIT = 150

# Contextual Understanding
context: List[str] = []
CONTEXT_SIZE = 10

def update_context(user_query: str, bot_response: str) -> None:
    
    print("Welcome! I'm here to help you with your Python programming questions.")
    """
    Update the context with the latest user query and bot response.
    """
    global context
    context.append(f"User: {user_query}")
    context.append(f"Bot: {bot_response}")
    while len(context) > CONTEXT_SIZE:
        context.pop(0)

def get_recent_context() -> str:
    """
    Retrieve the recent context for better understanding.
    """
    return "\n".join(context[-4:])

def check_code_db(query: str) -> Union[str, None]:
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

def handle_file_reading_request(query: str) -> str:
    file_path = query.split('"')[1]
    return read_file_content(file_path)


def handle_openai_response(query: str, context: List[str]) -> str:
    # Use the last 4 interactions for context
    full_prompt = "\n".join(context[-4:] + [f"User: {query}", "Bot:"])
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt=full_prompt,
            temperature=0.5,  # Reduce randomness
            max_tokens=TOKEN_LIMIT,
            top_p=1,
            frequency_penalty=0,  # Adjust penalties
            presence_penalty=0,
            stop=["\\n", "User:", "Bot:"]
        )
        response_text = response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
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
    if len(response_text.split()) > TOKEN_LIMIT:
        response_text = "My response seems too long. Would you like a more concise answer or should I clarify something specific?"

    return response_text

def chatbot_response(query: str) -> str:
    if not query.strip():
        return "It seems you didn't provide any input. Please ask a question or provide a command."
    global context
    context.append(query)

    if len(context) > CONTEXT_SIZE:
        context.pop(0)

    # Check for file reading requests
    if "C:\\" in query and ("read" in query or "open" in query or "type out" in query):
        return handle_file_reading_request(query)


# If no other conditions are met, use Dialogflow for general chatbot interactions

    if dialogflow_response := get_dialogflow_response(query):
        return dialogflow_response

    # Use Cloud Natural Language API for sentiment analysis
    try:
        sentiment_score, sentiment_magnitude = analyze_text(query)
        if  sentiment_score > 0.7:
            return "I'm glad to hear that!"
        elif sentiment_score < -0.7:
            return "I'm sorry to hear that. How can I assist you further?"
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

    # If neither Dialogflow nor sentiment analysis provides a clear response, use OpenAI API
    return handle_openai_response(query, get_recent_context())


