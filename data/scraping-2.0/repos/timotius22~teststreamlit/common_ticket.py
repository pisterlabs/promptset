import streamlit as st
import openai
import json

# Function to load prompts from the JSON file
def load_prompts():
    with open('prompts.json', 'r') as file:
        return json.load(file)

# Functions to format the output
def format_normal(ticket):
    # Convert HTML-like tags to readable text
    ticket = ticket.replace("<h1>", "").replace("</h1>", "\n")
    ticket = ticket.replace("<p>", "").replace("</p>", "\n\n")
    # Add more replacements as needed for other tags
    return ticket.strip()

def format_html(ticket):
    # Wrap the ticket in HTML body tags
    return f"<html><body>\n{ticket}\n</body></html>"

def format_jira(ticket):
    # Replace HTML tags with Jira Markup equivalents
    ticket = ticket.replace("<h1>", "h1. ").replace("</h1>", "\n")
    ticket = ticket.replace("<p>", "").replace("</p>", "\n\n")
    # Add more replacements as needed for other tags
    return ticket.strip()

# Function to apply selected format to ticket
def apply_format(ticket, format_selection):
    if format_selection == "Normal":
        return format_normal(ticket)
    elif format_selection == "HTML":
        return format_html(ticket)
    elif format_selection == "Jira Markup Language":
        return format_jira(ticket)
    return ticket  # Default to normal if no format is matched

# Main function to show ticket
def show_ticket():
    st.title("Product GPT")

    openai.api_key = st.session_state.get('api_key', st.secrets["openai"]["api_key"])

    ticket_type = st.selectbox("Select the ticket type:", ["Bug", "User Story", "Task", "Spike"])
    user_input = st.text_area("Write your ticket here:")

    format_selection = st.selectbox("Select the output format:", ["Normal", "HTML", "Jira Markup Language"])

    if st.button("Create Ticket") or 'ticket_content' in st.session_state:
        prompts = load_prompts()
        prompt_text = prompts.get(ticket_type, "")
        if not prompt_text:
            st.error(f"Could not find a prompt for ticket type: {ticket_type}")
            return

        if 'ticket_content' not in st.session_state or st.button("Create Ticket"):
            prompt = {"role": "user", "content": prompt_text + user_input}
            system_prompt = {"role": "system", "content": "You are an experienced product manager and an expert in writing tickets."}

            try:
                with st.spinner("Creating your ticket... This may take up to two minutes."):
                    response = openai.ChatCompletion.create(
                        model="gpt-4-1106-preview",
                        messages=[system_prompt, prompt]
                    )

                    # Check the response structure
                    if 'choices' not in response or not response['choices']:
                        st.error("Invalid response structure from OpenAI.")
                        st.json(response)  # Display the raw response for debugging
                        return

                    ticket = response.choices[0].get("message")

                    if ticket is None:
                        st.error("No ticket generated. The completion result was empty.")
                        return

                    if not isinstance(ticket, str):
                        st.error("Invalid ticket format.")
                        st.write(f"Received ticket data type: {type(ticket)}")  # Display the type of the received ticket for debugging
                        return

                    st.session_state['ticket_content'] = ticket
                    st.success("Ticket created successfully:")

            except openai.error.OpenAIError as openai_error:
                st.error(f"An error occurred with OpenAI: {openai_error}")
                return
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                return

        # Apply selected format and display
        formatted_ticket = apply_format(st.session_state['ticket_content'], format_selection)
        if format_selection == "HTML":
            st.code(formatted_ticket, language="html")
        elif format_selection == "Jira Markup Language":
            st.code(formatted_ticket, language="markup")
        else:
            st.text(formatted_ticket)

