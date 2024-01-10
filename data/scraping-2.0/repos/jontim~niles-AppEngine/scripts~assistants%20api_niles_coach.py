from openai import OpenAI
import shelve
from google.cloud import secretmanager
import time


# Function to access secret in Google Secret Manager
def access_secret_version(project_id, secret_id, version_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode('UTF-8')

# Get the OpenAI API key from Google Secret Manager
OPEN_AI_API_KEY = access_secret_version('api-call-niles', 'projects/857423205039/secrets/OPENAI_API_KEY', '1')

client = OpenAI(api_key=OPEN_AI_API_KEY)

# --------------------------------------------------------------
# Upload file
# --------------------------------------------------------------
# def upload_file(path):
    # Upload a file with an "assistants" purpose
#    file = client.files.create(file=open(path, "rb"), purpose="assistants")
#    return file


# file = upload_file("../data/airbnb-faq.pdf") # 


# --------------------------------------------------------------
# Create assistant
# --------------------------------------------------------------
# def create_assistant(file):
  #  """
   # You currently cannot set the temperature for Assistant via the API.
  #  """
  #  assistant = client.beta.assistants.create(
   #     name="WhatsApp AirBnb Assistant",
    #    instructions="You're a helpful WhatsApp assistant that can assist guests that are staying in our Paris AirBnb. Use your knowledge base to best respond to customer queries. If you don't know the answer, say simply that you cannot help with question and advice to contact the host directly. Be friendly and funny.",
     #   tools=[{"type": "retrieval"}],
      #  model="gpt-4-1106-preview",
       # file_ids=[file.id],

#)
    # return assistant


# assistant = create_assistant(file) #


# --------------------------------------------------------------
# Thread management
# --------------------------------------------------------------
def check_if_thread_exists(wa_id):
    with shelve.open("threads_db") as threads_shelf:
        return threads_shelf.get(wa_id, None)


def store_thread(wa_id, thread_id):
    with shelve.open("threads_db", writeback=True) as threads_shelf:
        threads_shelf[wa_id] = thread_id


# --------------------------------------------------------------
# Generate response
# --------------------------------------------------------------
def generate_response(message_body, wa_id, name):
    # Check if there is already a thread_id for the wa_id
    thread_id = check_if_thread_exists(wa_id)

    # If a thread doesn't exist, create one and store it
    if thread_id is None:
        print(f"Creating new thread for {name} with wa_id {wa_id}")
        thread = client.beta.threads.create()
        store_thread(wa_id, thread.id)
        thread_id = thread.id

    # Otherwise, retrieve the existing thread
    else:
        print(f"Retrieving existing thread for {name} with wa_id {wa_id}")
        thread = client.beta.threads.retrieve(thread_id)

    # Add message to thread
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_body,
    )

    # Run the assistant and get the new message
    new_message = run_assistant(thread)
    print(f"To {name}:", new_message)
    return new_message


# --------------------------------------------------------------
# Run assistant
# --------------------------------------------------------------
def run_assistant(thread):
    # Retrieve the Assistant
    assistant = client.beta.assistants.retrieve("asst_rLgLGT2ZktGeTRDqYZhrdC97")

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # Wait for completion
    while run.status != "completed":
        # Be nice to the API
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    # Retrieve the Messages
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    new_message = messages.data[0].content[0].text.value
    print(f"Generated message: {new_message}")
    return new_message


# --------------------------------------------------------------
# Test assistant
# --------------------------------------------------------------

# new_message = generate_response("How do I apply emotional regulation when I'm really stressed in a conversation?", "911", "Lisa")
