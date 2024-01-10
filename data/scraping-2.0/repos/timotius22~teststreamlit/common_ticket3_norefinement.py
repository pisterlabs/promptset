import streamlit as st
import openai
import json

# Function to load prompts from the JSON file
def load_prompts():
    """Load prompts from a JSON file."""
    with open('prompts.json', 'r') as file:
        return json.load(file)

def format_normal(ticket):
    """Format the ticket in a normal, readable text format, similar to provided example."""
    ticket = ticket.replace("<h1>", "").replace("</h1>", "\n")
    ticket = ticket.replace("<h2>", "").replace("</h2>", "\n")
    ticket = ticket.replace("<br>", "\n")
    ticket = ticket.replace("<p>", "").replace("</p>", "\n")
    ticket = ticket.replace("<ol>", "").replace("</ol>", "\n")
    ticket = ticket.replace("<ul>", "").replace("</ul>", "\n")
    ticket = ticket.replace("<li>", " - ").replace("</li>", "\n")
    ticket = ticket.replace("<strong>", "").replace("</strong>", "")
    return ticket.strip()

def format_html(ticket):
    """Format the ticket in HTML, based on the provided example."""
    # Escape special characters here if needed
    return f"<html><body>\n{ticket}\n</body></html>"

def format_jira(ticket):
    """Format the ticket in Jira Markup Language, according to the provided example."""
    ticket = ticket.replace("<h1>", "h1. ").replace("</h1>", "\n")
    ticket = ticket.replace("<h2>", "h2. ").replace("</h2>", "\n")
    ticket = ticket.replace("<br>", "\n")
    ticket = ticket.replace("<p>", "").replace("</p>", "\n")
    ticket = ticket.replace("<ol>", "").replace("</ol>", "\n")
    ticket = ticket.replace("<ul>", "").replace("</ul>", "\n")
    ticket = ticket.replace("<li>", "- ").replace("</li>", "\n")
    ticket = ticket.replace("<strong>", "*").replace("</strong>", "*")
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


# Main function to show and create ticket
def show_ticket():
    """Main function to display and create tickets."""
    st.title("Product GPT")

    # Setting up OpenAI API key
    openai.api_key = st.session_state.get('api_key', st.secrets["openai"]["api_key"])

    ticket_type = st.selectbox("Select the ticket type:", ["Bug", "User Story", "Task", "Spike"])
    user_input = st.text_area("Write your ticket here:")
    format_selection = st.selectbox("Select the output format:", ["Normal", "HTML", "Jira Markup Language"], key='format_selector')

    # Create a placeholder for the formatted ticket
    formatted_ticket_placeholder = st.empty()

    if st.button("Create/Update Ticket"):
        with st.spinner('Creating/Updating ticket...'):
            prompts = load_prompts()
            prompt_text = prompts.get(ticket_type, "")
            if not prompt_text:
                st.error(f"Could not find a prompt for ticket type: {ticket_type}")
                return

            prompt = {"role": "user", "content": prompt_text + user_input}
            system_prompt = {"role": "system", "content": "You are an experienced product manager and an expert in writing tickets. You only ever reply with the ticket, and no extra fields. You follow the html template."}
            st.session_state['conversation_history'] = [system_prompt, prompt]

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=st.session_state['conversation_history']
                )
                gpt_response = response.choices[0].message["content"]
                if gpt_response:
                    st.session_state['original_response'] = gpt_response
                    updated_formatted_ticket = apply_format(gpt_response, format_selection)
                    formatted_ticket_placeholder.text_area("Formatted Ticket", updated_formatted_ticket, height=300, key='formatted_ticket')
            except openai.error.OpenAIError as openai_error:
                st.error(f"An error occurred with OpenAI: {openai_error}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        # Check if there's an existing ticket to format
        if 'original_response' in st.session_state and 'format_selector' in st.session_state:
            updated_formatted_ticket = apply_format(st.session_state['original_response'], st.session_state['format_selector'])
            formatted_ticket_placeholder.text_area("Formatted Ticket", updated_formatted_ticket, height=300, key='formatted_ticket')
