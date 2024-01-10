import streamlit as st
import requests
import openai

# Set OpenAI API key
openai.api_key = 'Your OpenAi Key'

# Function to get access token
def get_access_token(client_id, client_secret, tenant_id):
    url = f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': 'https://graph.microsoft.com/.default',
        'grant_type': 'client_credentials'
    }
    response = requests.post(url, headers=headers, data=data)
    response_json = response.json()
    return response_json['access_token']

# Function to get event logs
def get_event_logs(access_token):
    url = 'https://graph.microsoft.com/v1.0/security/alerts?$top=8'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    return response.json()

# Function to analyze text
def analyze_text(text, event_logs):
    # Convert the event logs to a string
    event_logs_str = str(event_logs)

    # Create a list of messages
    messages = [
        {"role": "system", "content": event_logs_str},
        {"role": "user", "content": text}
    ]

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=messages
    )
    return response['choices'][0]['message']['content']

# Streamlit app
def main():
    st.title('LogSpotLight-SecurityEvents')

    # Get access token
    client_id = 'App Client Id'
    client_secret = 'App Client Secret'
    tenant_id = 'Tenant Id'
    access_token = get_access_token(client_id, client_secret, tenant_id)

    # Get event logs
    event_logs = get_event_logs(access_token)

    # Display event logs
    st.write('Event Logs:', event_logs)

    # User input
    user_input = st.text_input('Enter your query:')

    if user_input:
        # Analyze user input
        analysis = analyze_text(user_input, event_logs)

        # Display analysis
        st.write('Analysis:', analysis)

if __name__ == '__main__':
    main()



