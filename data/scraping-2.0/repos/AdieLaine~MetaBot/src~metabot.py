import openai
import streamlit as st
import time
from datetime import datetime
import pandas as pd
import gzip
import statistics
from typing import List

# Set OpenAI API key
openai.api_key = "set-your-api-key-here"

# Set OpenAI API key using Streamlit secrets management
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit configurations
st.set_page_config(
    page_title="MetaBot",
    page_icon="🗣️",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/AdieLaine/MetaBot/',
        'Report a bug': 'https://github.com/AdieLaine/MetaBot/',
        'About': """
            # MetaBot
            MetaBot is an advanced ChatBot that leverages the power of multiple OpenAI models. Using a unique 'model sliding' approach, it intelligently selects the most suitable model for a given task based on the user's prompt. Each task type is associated with specific keywords that link to the most effective model. If no keywords match, a default model is used.
            https://github.com/AdieLaine/MetaBot/
        """
    }
)

st.markdown('<h1 style="text-align: center; color: seaGreen; margin-top: -70px;">MetaBot</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;"><strong>A Multi-Model AI Assistant</strong></h3>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

@st.cache_resource
def part_of_day(hour):
    """
    Determines the part of the day (morning, afternoon, evening) based on the hour.

    Args:
        hour (int): The current hour.

    Returns:
        str: The part of the day.
    """
    return (
        "morning" if 5 <= hour <= 11
        else
        "afternoon" if 12 <= hour <= 17
        else
        "evening"
    )

def create_compressor(compressor='gzip'):
    """
    Initialises and returns gzip compression. The return compressor is used to compress and calculate the size of 
    the text for Normalized Compression Distance. gzip is used to compress the text.

    Args:
        compressor (str): the name of the compressor to be created which is 'gzip' by default.

    Returns:
        compressor (gzip): the gzip compressor.
        
    Raises:
        RuntimeError: if a compressor other than gzip is requested.
    """
    if compressor != 'gzip':
        raise RuntimeError("Unsupported compressor")
    print("Using gzip compression.")
    return gzip

def compress_text(compressor, text):
    """
    Compress the given text using the gzip compression created using 'create_compressor()'. 
    The function prints the length of the original text and the length of the compressed text.

    Args:
        compressor (gzip): gzip compressor.
        text (str): the text to be compressed.

    Returns:
        compressed (gzip): the compressed text.
    """
    compressed = compressor.compress(text.encode("utf-8"))
    print(f"Original size of '{text}' is {len(text)}")
    print(f"Compressed size of '{text}' is {len(compressed)}")
    return compressed


def compute_distance(text_1: str, text_2: str, compressor):
    """
    Compute the Normalized Compression Distance between two texts. Compute the normalized distance by minizinc the size of 
    compressed two texts and the compressed individual texts.

    Args:
        text_1 (str): the first text.
        text_2 (str): the second text.
        compressor (gzip): gzip compressor.

    Returns:
        normalized_distance (float): the computed normalized distance between text1 and text2.
    """
    combined = f"{text_1} {text_2}"
    compressed_size_combined = len(compress_text(compressor, combined))
    compressed_size_1 = len(compress_text(compressor, text_1))
    compressed_size_2 = len(compress_text(compressor, text_2))
    normalized_distance = (
        compressed_size_combined - min(compressed_size_1, compressed_size_2)
    ) / max(compressed_size_1, compressed_size_2)
    print(f"Normalized distance between '{text_1}' and '{text_2}' is {normalized_distance}")
    return normalized_distance

def predict_class(test_entry: str, training_data: List[str], training_labels: List[str], n_neighbors: int, compressor) -> str:
    """
    Predict the class of the input text based on the trained data.

    Args:
        test_entry (str): the input text.
        training_data (List[str]): the training data.
        training_labels (List[str]): the labels for the training data.
        n_neighbors (int): the number of nearest neighbors.
        compressor (gzip): gzip compressor.

    Returns:
        result (str): the predicted class.
    """
    distance_from_training = [
        compute_distance(test_entry, train_entry, compressor)
        for train_entry in training_data
    ]
    sorted_indices = sorted(
        range(len(distance_from_training)),
        key=lambda i: distance_from_training[i],
    )
    top_n_class = [
        training_labels[i] for i in sorted_indices[: n_neighbors]
    ]
    return statistics.mode(top_n_class)

def predict(X: List[str], training_data: List[str], training_labels: List[str], n_neighbors: int, compressor) -> List[str]:
    """
    Predict the classes for multiple texts.

    Args:
        X (List[str]): the input texts.
        training_data (List[str]): the training data.
        training_labels (List[str]): the labels for the training data.
        n_neighbors (int): the number of nearest neighbors.
        compressor (gzip): gzip compressor.

    Returns:
        result (List[str]): the predicted classes.
    """
    return [predict_class(test_entry, training_data, training_labels, n_neighbors, compressor) for test_entry in X]


# List for training data and labels
training_data = [
    "Write me a python script",
    "Help me with my English essay",
    "Tell me a fantasy story",
    "Create a social media post",
    "Find information on AI",
    "Help with sales strategy",
    "Need advice on learning"
]
training_labels = ["code", "essay", "story", "social media", "research", "sales", "advice"]

# Incorporate these functions into the generate_response function
def generate_response(user_input):
    """
    Generate a response by predicting the class of the input text using the trained data.

    Args:
        user_input (str): the input text from the user

    Returns:
        response (str): the generated response.
    """
    response = predict([user_input], training_data, training_labels, n_neighbors=10, compressor=create_compressor('gzip'))[0]
    return response

@st.cache_data
def meta_model(prompt, model_roles):
    """
    Determines the most appropriate model to use based on the user's prompt.

    Args:
        prompt (str): The task description provided by the user.
        model_roles (dict): A dictionary mapping keywords to models, roles, and examples.

    Returns:
        tuple: The selected model, the role of the assistant, and the example.
    """
    # Convert the prompt to lower case for case insensitive matching
    prompt = prompt.lower()

    # Iterate over the dictionary to find the appropriate model, role, and example
    for keywords, (model, role, example) in model_roles.items():
        if any(keyword in prompt for keyword in keywords):
            return model, role, example

    # If no keywords match, default to the base model and a general role
    return "gpt-3.5-turbo", 'A helpful assistant.', 'I can assist with various tasks.'

# Define the model_roles dictionary
model_roles = {
    ("code", "program", "script"): ("gpt-3.5-turbo", 'You are a programming assistant that logically weaves code together.', 'Code a Streamlit program that is helpful.'),
    ("essay", "paper", "report", "article"): ("gpt-4", 'You are a writing assistant that excels at creating amazing written material.', 'Write an technical essay on AI.'),
    ("story", "narrative", "tale", "fable"): ("gpt-3.5-turbo", 'You are a storyteller assistant that can weave intricate and compelling stories.', 'Tell me an epic story.'),
    ("social media", "post", "content", "engaging"): ("gpt-4", 'You are a social media assistant skilled in creating engaging and creative content for social media.', 'Make an engaging social media post with exciting news.'),
    ("research", "investigate", "information"): ("gpt-4", 'You are a research assistant skilled at finding and synthesizing information.', 'Research AI related topics.'),
    ("sales", "market", "promote"): ("gpt-3.5-turbo", 'You are a sales assistant who can help with marketing and promotion strategies.', 'Create a sales strategy plan using a table chart.'),
    ("advice", "recommend", "suggest"): ("gpt-3.5-turbo", 'You are an advice assistant who can provide recommendations and suggestions.', 'I need some advice on how to learn a new skill.')
}

def add_message(role, content):
    """
    Generates a new message in the required format. If the message is a system message,
    it generates a dynamic greeting based on the time of day with keywords.

    Args:
        role (str): The role of the message ("system", "user", or "assistant").
        content (str): The content of the message.

    Returns:
        dict: A dictionary representing the message.
    """
    if role == "assistant" and content == "greeting":
        current_hour = datetime.now().hour
        day_part = part_of_day(current_hour)
        content = f'Good {day_part}! I\'m your AI assistant. I can help you with a variety of tasks.'
    elif content.strip() == "!":
        content = "helpme"
    return {"role": role, "content": content}

def generate_response(model, messages, message_placeholder):
    """
    Generates a response using the selected OpenAI model.

    This function uses the OpenAI API to generate a response based on the conversation history. 
    If the selected model is a fine-tuned model, it uses the `openai.Completion.create` method. 
    For other models, it uses the `openai.ChatCompletion.create` method with streaming enabled for a realistic typing effect.

    Args:
        model (str): The model to use for generation. This is either "fine-tuned-model" or the name of an OpenAI base model.
        messages (list): The list of message dicts for the conversation history. Each dict has a "role" key and a "content" key.
        message_placeholder (streamlit.delta_generator.DeltaGenerator): An empty Streamlit container for the message to be generated.

    Returns:
        str: The generated response.

    Raises:
        OpenAIError: If there is an error with the OpenAI API request.
    """
    try:
        print(f"Using {model} for response generation.")  # Print the model being used

        if model == "fine-tuned-model":
            response = openai.Completion.create(
                model=model,
                prompt=messages[-1]["content"],
                max_tokens=60
            )
            return response.choices[0].text.strip()
        else:
            accumulated_response = ""
            # Use streaming approach for standard models
            for response in openai.ChatCompletion.create(
                model=model,
                messages=messages,
                stream=True,
            ):
                chunk = response.choices[0].delta.get("content", "")
                accumulated_response += chunk
                for word in chunk.split():
                    message_placeholder.markdown(accumulated_response + " " + "▌", unsafe_allow_html=True)
                    time.sleep(0.01)  # delay between words for realistic typing effect
            message_placeholder.markdown(accumulated_response, unsafe_allow_html=True)
            return accumulated_response
    except openai.OpenAIError as e:
        print(f"Error occurred: {e}")  # Print the error message
        return "I'm sorry, but I'm unable to generate a response at the moment."

@st.cache_data
def display_model_table():
    """
    Creates and displays a table of models and their associated keywords along with example usages using Streamlit's st.table function.

    Returns:
        None
    """
    # Create a DataFrame for the table
    model_table = pd.DataFrame({
        "Model": ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo"],
        "First Keyword": ["code", "essay", "story", "social media", "research", "sales", "advice"],
        "Other Keywords": [
            "program, script",
            "paper, report, article",
            "narrative, tale, fable",
            "post, content, engaging",
            "investigate, information",
            "market, promote",
            "recommend, suggest"
        ],
        "Role": [
            "Assistant specialized in generating logical code in any programming language", 
            "Assistant that excels at writing well-structured and grammatically correct text",
            "Assistant that can weave intricate and compelling stories", 
            "Assistant skilled in creating engaging and creative content for social media",
            "Assistant skilled at finding and analyzing information",
            "Assistant who can help with marketing and promotion strategies",
            "Assistant who can provide advice and recommendations"
        ],
        "Example": [
            "Code a Streamlit program that is helpful.",
            "Write an technical essay on AI.",
            "Tell me an epic story.",
            "Make an engaging social media post with exciting news.",
            "Research AI related topics.",
            "Create a sales strategy plan using a table chart.",
            "I need some advice on how to learn a new skill."
        ]
    })

    # Remove the index
    model_table.set_index("Model", inplace=True)

    # Display the table
    st.table(model_table)

# Initialize the session state if not already done
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
if "messages" not in st.session_state:
    current_hour = datetime.now().hour
    day_part = part_of_day(current_hour)
    greeting_message = f"Good {day_part}! I\'m your AI assistant. I can help you with a variety of tasks."
    st.session_state.messages = [add_message("assistant", "greeting")]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Capture the user's input and generate the AI's response
if prompt := st.chat_input("Enter your prompt or 'helpme' or '!' for commands."):
    model, role, example = meta_model(prompt, model_roles)
    st.session_state["openai_model"] = model  # Update the model in the session state

    if prompt.lower().strip() in ["helpme", "!"]:
        table_html = "<style>.model-table {width: 100%; border-collapse: collapse;}.model-table th, .model-table td {border: 1px solid #dddddd; padding: 8px; text-align: left;}.model-table tr:nth-child(even) {background-color: #slategray;}.keyword {font-weight: bold;}.model-name {color: seaGreen; font-weight: bold;}.example-word {color: CornflowerBlue; font-weight: bold;}</style>"
        table_html += "<table class=\"model-table\"><tr><th>Model</th><th>First Keyword</th><th>Other Keywords</th><th>Role</th><th>Example</th></tr>"
        for keywords, (model, role, example) in model_roles.items():
            first_keyword = keywords[0]
            other_keywords = ", ".join(keywords[1:])
            table_html += f"<tr><td class=\"model-name\">{model}</td><td class=\"keyword\">{first_keyword}</td><td>{other_keywords}</td><td>{role}</td><td class=\"example-word\">{example}</td></tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.session_state.messages.append(add_message("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Model selection explanation
            model_explanations = {
                "gpt-3.5-turbo": "This task used GPT-3.5-Turbo. This model is optimized for tasks related to Programming, Storytelling, Strategy and Advice.",
                "gpt-4": "This task used GPT-4. This model excels at tasks related to Writing, Social Media Content Creation and Research."
            }
            full_response = generate_response(model, st.session_state.messages, message_placeholder)
            message_placeholder.markdown(full_response)
            # For demonstration purposes, we print the model explanation
            st.info(model_explanations[model])
        st.session_state.messages.append(add_message("assistant", full_response))
#samqwan