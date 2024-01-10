from openai import OpenAI
import os, json
import time

def create_assistant(client):
    assistant_file_path = 'assistant.json'

    if os.path.exists(assistant_file_path):
        with open(assistant_file_path, 'r') as file:
            assistant_data = json.load(file)
            assistant_id = assistant_data['assistant_id']
            print("Loaded existing assistant ID.")
    else:
        file = client.files.create(file=open("knowledges_small_1.json", "rb"),
                                   purpose='assistants')

        assistant = client.beta.assistants.create(instructions="""
Please solve the following problem with a meticulous and precise approach, characteristic of someone with OCD who is committed to accuracy in every response. Begin by carefully reading the problem, paying attention to all details. Derive an equation based on the relevant mathematical or scientific principles inherent in the problem. Once you have the equation, use the options provided in the problem, like 'The monthly rent of a shop of dimensions 20 feet × 18 feet is Rs. 1440. What is the annual rent per square foot of the shop? Options: a) 48, b) 56, c) 68, d) 87, e) 92', to substitute into this equation.

Carefully analyze the problem, considering all potential scenarios. Solve the problem step by step, substituting the options into your derived equation. Review your solution meticulously for errors or oversights. Compare your results with the given options and select the one that accurately fits your calculated answer. If uncertain or if the problem is incorrect with no correct options, thoughtfully choose the most plausible answer based on the available information. Please avoid using symbols like double-quotes, dollar signs, commas, or exclamation marks in your answers.
          """,
              model="gpt-4-1106-preview",
              tools=[{
                "type": "retrieval"
              }],
              file_ids=[file.id])

        with open(assistant_file_path, 'w') as file:
            json.dump({'assistant_id': assistant.id}, file)
            print("Created a new assistant and saved the ID.")

        assistant_id = assistant.id

    return assistant_id


# Key
api_key = "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"
client = OpenAI(api_key=api_key)


# Create new assistant or load existing
assistant_id = create_assistant(client)

def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


def create_thread_and_run(user_input):
    thread = client.beta.threads.create()
    run = submit_message(assistant_id, thread, user_input)
    return thread, run
def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


# Waiting in a loop
def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


thread, run = create_thread_and_run("assume all pieces of rope are equal . if 44 pieces of rope measure a feet , how long would b pieces of rope be in inches ?")
run = wait_on_run(run, thread)
pretty_print(get_response(thread))