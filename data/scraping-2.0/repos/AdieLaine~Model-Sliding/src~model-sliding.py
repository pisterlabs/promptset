import openai
import streamlit as st
import time
from datetime import datetime
import pandas as pd

# Set OpenAI API key
openai.api_key = "set-your-api-key-here"

# Set OpenAI API key using Streamlit secrets management
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit configurations
st.set_page_config(
    page_title="Model Sliding",
    page_icon="🛝",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/AdieLaine/Model-Sliding/',
        'Report a bug': 'https://github.com/AdieLaine/Model-Sliding/',
        'About': """
            # Model Sliding
            This application demonstrates 'model sliding', a method of logic that selects the best OpenAI model for a given task based on the user's prompt. Each task type is associated with specific keywords that link to the most suitable model. If no keywords match, a default model is used.
            https://github.com/AdieLaine/Model-Sliding/
        """
    }
)

st.markdown('<h1 style="text-align: center; color: seaGreen; margin-top: -70px;">Model Sliding</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;"><strong>AI Model Automation</strong></h3>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)


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

@st.cache_data
def meta_model(prompt, model_roles):
    """
    Determines the most appropriate model to use based on the user's prompt.

    Args:
        prompt (str): The task description provided by the user.
        model_roles (dict): A dictionary mapping keywords to models and roles.

    Returns:
        tuple: The selected model and the role of the assistant.
    """
    # Convert the prompt to lower case for case insensitive matching
    prompt = prompt.lower()

    # Iterate over the dictionary to find the appropriate model and role
    for keywords, (model, role) in model_roles.items():
        if any(keyword in prompt for keyword in keywords):
            return model, role

    # If no keywords match, default to the base model and a general role
    return "gpt-3.5-turbo", 'A helpful assistant.'


# Define the model_roles dictionary
model_roles = {
    ("code", "programming", "algorithm"): ("gpt-3.5-turbo", 'You are an programming assistant that logically weaves code together.'),
    ("essay", "paper", "report", "article"): ("gpt-4", 'You are an writing assistant that excels at creating amazing written material.'),
    ("story", "narrative", "tale", "fable"): ("gpt-3.5-turbo", 'You are an storyeller assistant that can weave intricate and compelling stories.'),
    ("social media", "post", "content", "engaging"): ("gpt-4", 'You are an social media assistant skilled in creating engaging and creative content for social media.')
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

    Args:
        model (str): The model to use for generation.
        messages (list): The list of message dicts for the conversation history.
        message_placeholder (streamlit.delta_generator.DeltaGenerator): An empty Streamlit container for the message to be generated.

    Returns:
        str: The generated response.
    """
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

@st.cache_data
def display_model_table():
    """
    Creates and displays a table of models and their associated keywords along with example usages using Streamlit's st.table function.

    Returns:
        None
    """
    # Create a DataFrame for the table
    model_table = pd.DataFrame({
        "Model": ["gpt-3.5-turbo", "gpt-3.5-turbo", "gpt-4", "gpt-4"],
        "First Keyword": ["code", "story", "essay", "social media"],
        "Other Keywords": [
            "programming, algorithm", 
            "narrative, tale, fable", 
            "paper, report, article", 
            "post, content, engaging"
        ],
        "Role": [
            "Assistant specialized in generating Python code", 
            "Assistant that can weave intricate and compelling stories", 
            "Assistant that excels at writing well-structured and grammatically correct text", 
            "Assistant skilled in creating engaging and creative content for social media"
        ],
        "Example": [
            "Create a Streamlit app with docstrings.",
            "Tell me a story about AI.",
            "Write an technical essay on LLM's.",
            "Craft a social media post with exciting news."
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
    model, role = meta_model(prompt, model_roles)
    if prompt.lower().strip() in ["helpme", "!"]:
        st.markdown(
            """
            <style>
                .model-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .model-table th, .model-table td {
                    border: 1px solid #dddddd;
                    padding: 8px;
                    text-align: left;
                }
                .model-table tr:nth-child(even) {
                    background-color: #slategray;
                }
                .keyword {
                    font-weight: bold;
                }
                .model-name {
                    color: seaGreen;
                    font-weight: bold;
                }
                .example-word {
                    color: CornflowerBlue;
                    font-weight: bold;
                }
            </style>
            <table class="model-table">
                <tr>
                    <th>Model</th>
                    <th>First Keyword</th>
                    <th>Other Keywords</th>
                    <th>Role</th>
                </tr>
                <tr>
                    <td class="model-name">GPT-3.5-Turbo</td>
                    <td class="keyword">code</td>
                    <td>programming, algorithm</td>
                    <td>Assistant specialized in generating Python code</td>
                </tr>
                <tr>
                    <td colspan="4"><span class="example-word">Example:</span> Code a Streamlit app with docstrings.</td>
                </tr>
                <tr>
                    <td class="model-name">GPT-3.5-Turbo</td>
                    <td class="keyword">story</td>
                    <td>narrative, tale, fable</td>
                    <td>Assistant that can weave intricate and compelling stories</td>
                </tr>
                <tr>
                    <td colspan="4"><span class="example-word">Example:</span> Tell me a <span class="keyword">story</span>  about AI.</td>
                </tr>
                <tr>
                    <td class="model-name">GPT-4</td>
                    <td class="keyword">essay</td>
                    <td>paper, report, article</td>
                    <td>Assistant that excels at writing well-structured and grammatically correct text</td>
                </tr>
                <tr>
                    <td colspan="4"><span class="example-word">Example:</span> Write an technical <span class="keyword">essay</span>  on LLM's.</td>
                </tr>
                <tr>
                    <td class="model-name">GPT-4</td>
                    <td class="keyword">social media</td>
                    <td>post, content, engaging</td>
                    <td>Assistant skilled in creating engaging and creative content for social media</td>
                </tr>
                <tr>
                    <td colspan="4"><span class="example-word">Example:</span> Craft an <span class="keyword">engaging social media</span>  post with exciting news.</td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True
        )

    else:
        st.session_state.messages.append(add_message("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Model selection explanation
            model_explanations = {
                "gpt-3.5-turbo": "The 'GPT-3.5-Turbo' model was chosen because it is optimized for generating application code and storytelling.",
                "gpt-4": "The 'GPT-4' model was chosen because it excels at writing well-structured and grammatically correct text, and creating engaging and creative content for social media."
            }
            full_response = generate_response(model, st.session_state.messages, message_placeholder)
            message_placeholder.markdown(full_response)
            # For demonstration purposes, we print the model explanation
            st.info(model_explanations[model])
        st.session_state.messages.append(add_message("assistant", full_response))
#iapjiw