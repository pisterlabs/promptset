import streamlit as st
import openai
import json

# Function to load prompts from the JSON file
def load_prompts():
    """Load prompts from a JSON file."""
    with open('prompts.json', 'r') as file:
        return json.load(file)

# Functions to format the output
def format_normal(ticket):
    """Format the ticket in a normal, readable text format, similar to provided example."""
    ticket = ticket.replace("<h1>", "").replace("</h1>", "")
    ticket = ticket.replace("<br>", "")
    ticket = ticket.replace("<p>", "").replace("</p>", "")
    ticket = ticket.replace("<ol>", "").replace("</ol>", "")
    ticket = ticket.replace("<ul>", "").replace("</ul>", "")
    ticket = ticket.replace("<li>", " - ").replace("</li>", "\n")
    return ticket.strip()

def format_html(ticket):
    """Format the ticket in HTML, based on the provided example."""
    return f"<html><body>\n{ticket}\n</body></html>"

def format_jira(ticket):
    """Format the ticket in Jira Markup Language, according to the provided example."""
    ticket = ticket.replace("<h1>", "h1. ").replace("</h1>", "")
    ticket = ticket.replace("<br>", "")
    ticket = ticket.replace("<p>", "").replace("</p>", "")
    ticket = ticket.replace("<ol>", "").replace("</ol>", "")
    ticket = ticket.replace("<ul>", "").replace("</ul>", "")
    ticket = ticket.replace("<li>", "- ").replace("</li>", "")
    return ticket.strip()

# Function to apply selected format to ticket
def apply_format(ticket, format_selection):
    """Apply the selected format to the ticket."""
    if format_selection == "Normal":
        return format_normal(ticket)
    elif format_selection == "HTML":
        return format_html(ticket)
    elif format_selection == "Jira Markup Language":
        return format_jira(ticket)
    return ticket  # Default to normal if no format is matched

# Function to send message to OpenAI and get response
def chat_with_openai(message, history):
    """Send a message to OpenAI and get the response."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=history + [message]
        )
        return response.choices[0].message["content"]
    except openai.error.OpenAIError as openai_error:
        st.error(f"An error occurred with OpenAI: {openai_error}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


# Function for refining the ticket

def refine_ticket_logic(refine_input, format_selection, history):
    """Refines the ticket based on user input."""
    with st.spinner('Refining ticket...'):
        user_message = {"role": "user", "content": refine_input}
        history.append(user_message)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=history
            )
            gpt_response = response.choices[0].message["content"]
            if gpt_response:
                history.append({"role": "system", "content": gpt_response})
                return apply_format(gpt_response, format_selection)
        except openai.error.OpenAIError as openai_error:
            st.error(f"An error occurred with OpenAI: {openai_error}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    return None
