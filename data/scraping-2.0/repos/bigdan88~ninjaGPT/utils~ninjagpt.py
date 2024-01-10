import openai
import time

# Initialize OpenAI client with your API key
client = openai.Client(api_key='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')
# openai.api_key = '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'

# Function to create the ninjaGPT assistant
def create_ninjaGPT(client):
    assistant = client.beta.assistants.create(
        name="ninjaGPT",
        instructions="You are an assistant inside a cooking gadget. Provide cooking advice.",
        tools=[],
        model="gpt-4-1106-preview"  # Assuming you are using the GPT-4 model
    )
    return assistant

# Function to create a thread for a new conversation
def create_thread(client):
    thread = client.beta.threads.create()
    return thread

# Function to send a message to ninjaGPT and receive a response
def ask_ninjaGPT(client, thread_id, assistant_id, question):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=question
    )
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    return run

# Function to update gadget parameters based on ninjaGPT's advice
def update_gadget_parameters(parameter_updates):
    # Implement this function to interface with your cooking gadget
    # Adjust temperature, time settings, etc., based on the assistant's advice
    pass

# Function to extract parameters from the assistant's response
def extract_parameters_from_response(response):
    # Implement logic to parse response for cooking parameter adjustments
    # Return a dictionary or similar structure with the extracted parameters
    pass

# Function to wait for a run to complete and then return its status
def wait_for_run_completion(client, thread_id, run_id):
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run_status.status == 'completed':  # Access status attribute directly
            return run_status
        time.sleep(1)  # Wait for 1 second before checking again


# Function to get all messages from a thread, including the assistant's responses
def get_thread_messages(client, thread_id):
    messages_page = client.beta.threads.messages.list(thread_id=thread_id)
    return messages_page.data if hasattr(messages_page, 'data') else []

if __name__ == "__main__":
    ninjaGPT = create_ninjaGPT(client)
    thread = create_thread(client)

    # Example question
    question = "How long should I bake chicken at 350 degrees Fahrenheit?"
    run = ask_ninjaGPT(client, thread.id, ninjaGPT.id, question)

    # Wait for the run to complete
    completed_run = wait_for_run_completion(client, thread.id, run.id)

    # Retrieve and print all messages from the thread
    messages = get_thread_messages(client, thread.id)
    for message in messages:
        print(f"Role: {message.role}, Content: {message.content[0].text.value}")