from openai import OpenAI
import time
import requests
import json

client = OpenAI(api_key="")

def upload_image(image_path):
    # Your OpenAI API key
    api_key = ""

    # The API endpoint for uploading files to OpenAI
    url = 'https://api.openai.com/v1/files'

    # Prepare the headers for authentication
    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    # Prepare the file data for upload
    files = {
        'file': open(image_path, 'rb'),
        'purpose': (None, 'assistants')
    }

    # Make the POST request to upload the file
    response = requests.post(url, headers=headers, files=files)

    # Close the file as we no longer need to read it
    files['file'].close()

    # Check the response
    if response.status_code == 200:
        print("Image uploaded successfully.")
        # Load the text into a Python dictionary
        json_object = json.loads(response.text)
        return json_object
    else:
        print("Failed to upload the image.")
        print("Status Code:", response.status_code)
        print("Response:", response.text)

response_json = upload_image("images/81f8557e-9fe3-475b-a7cd-08b57e9f4bb9-1702810465.987684.pdf")

FILE_ID = [response_json['id']]

# Create an assistant using the file ID
assistant = client.beta.assistants.create(
    instructions="You are a data analyst who examines the data in the charts presented as PDF files.",
    model="gpt-4-1106-preview", # "gpt-3.5-turbo", "gpt-4-1106-preview"
    tools=[{"type": "code_interpreter"},{"type": "retrieval"}],
    file_ids=FILE_ID
)

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Examine this PDF file and explain what is show on this chart.",
    file_ids=FILE_ID
)

run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# Poll the status until it is completed
while run.status != 'completed':
    time.sleep(5)  # Wait for 5 seconds before checking again
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    print(run.status)


print("Run completed. Status:", run.status)

# Fetch the messages
messages = client.beta.threads.messages.list(
    thread_id=thread.id
)

# Extracting assistant's message from the list of messages
assistant_messages = [msg.content for msg in messages.data if msg.role == 'assistant']


if assistant_messages:
    assistant_content = assistant_messages[0][0].text.value
    print("Assistant's Message Content:")
    print(assistant_content)
else:
    print("No assistant message found in the response.")

#image_descr = openai_inst.examine_file("file-EYEg3xBhCGKAMUGb5BQlE9Zn")
#print(assistant)

exit()