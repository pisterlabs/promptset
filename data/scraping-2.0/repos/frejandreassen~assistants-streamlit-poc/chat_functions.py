import streamlit as st
from openai import OpenAI
import time 
import json
from supabase import create_client, Client

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai_api_key"])  # Replace with your actual OpenAI API key
supabase_url = "https://yvpgmchgoiojubbdtvbc.supabase.co"  # Replace with your Supabase project URL
supabase_key = st.secrets["supabase_api_key"]  # Replace with your Supabase service role secret key
supabase: Client = create_client(supabase_url, supabase_key)

assistant_id = "asst_hPinVONMTZtTSNQRP5Vyg2vj"

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"

def search_for_falkenberg_info(text):
    print(text)
    embedding = get_embedding_for_text(text)
    similar_documents = fetch_similar_documents(embedding)
    return json.dumps(similar_documents)

# Function to fetch the most similar documents from Supabase
def fetch_similar_documents(query_embedding, match_threshold=0.78, match_count=3):
    data = supabase.rpc('match_documents', {
        'query_embedding': query_embedding,
        'match_threshold': match_threshold,
        'match_count': match_count,
    }).execute()
    return data.data

# Function to generate embeddings for user input
def get_embedding_for_text(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL, 
        input=text
    )
    return response.data[0].embedding



def get_weather(location, unit="celsius"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": location, "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": location, "temperature": "72", "unit": "fahrenheit"})
    else:
        return json.dumps({"location": location, "temperature": "22", "unit": "celsius"})



# Initialize the messages state if not already present
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Create a new thread for the conversation if not already present
if 'thread' not in st.session_state:
    st.session_state.thread = client.beta.threads.create()
thread = st.session_state.thread

st.write(thread.id)
st.title("Chat with functions")

# Display the chat history
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    with st.chat_message(role):
        st.markdown(content)

# Handle user input
user_input = st.chat_input("Ask about the weather:")
if user_input:
    
    # Add user input to messages and display
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add message to the thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    # Wait for the assistant to respond
    polls = 0
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        if run_status.status in ['completed', 'failed']:
            break

        # Call function
        if run_status.status == 'requires_action':
            print('REQUIRES ACTION!')
            available_functions = {
                "get_weather": get_weather,
                "search_for_falkenberg_info": search_for_falkenberg_info
            }
            for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)

                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=[
                        {
                            "tool_call_id": tool_call.id,
                            "output": function_response
                        }
                    ]
                )

        polls += 1
        if (polls > 20): break
        time.sleep(1.5)  # Sleep for half a second before checking again

    # Display the assistant's response
    if run_status.status == 'completed':
        # Get the messages here:
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        
        # Display each new message from the assistant
        message = messages.data[0]
        if message.role == "assistant":
            content = message.content[0].text.value
            st.session_state.messages.append({"role": "assistant", "content": content})
            with st.chat_message("assistant"):
                st.markdown(content)