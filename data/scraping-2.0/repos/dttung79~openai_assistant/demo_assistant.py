import time
import openai

def get_openai_key():
  with open('key.txt') as f:
    return f.read().strip()
  
# Create a OpenAI connection to access its threads and assistants
openai_client = openai.OpenAI(api_key=get_openai_key())
openai_threads = openai_client.beta.threads
openai_assistants = openai_client.beta.assistants

def main():
  assistant = create_assistant()
  thread = openai_threads.create()

  asking = True
  user_question = ask_question()

  while asking:
    run = ask_assistant(user_question, thread, assistant)
    assistant_response(thread, run)
    user_question = ask_question()
    asking = user_question.lower() != "quit"

    if not asking:
      print("Goodbye! I hope you will do well on math.")
  
def create_assistant():
  assistant = openai_assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. You will help user to answer math questions.",
    tools=[],
    model="gpt-4-1106-preview"
  )
  print("[Math Tutor]: Hello there, I'm your personal math tutor. Do you need any help?")

  return assistant

def ask_question():
  return input(f'[User]: ')

def ask_assistant(user_question, thread, assistant):
  # Pass in the user question into the existing thread
  openai_threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=user_question
  )

  # Use runs to wait for the assistant response 
  run = openai_threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    #instructions=f'Keep in mind that user has no knowledge of math, especially {user_question} so please explain the answer in detail and simple.'
  )
  
  is_running = True
  while is_running:
    run_status = openai_threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    is_running = run_status.status != "completed"
    time.sleep(1)

  return run

def assistant_response(thread, run):
  # Get the messages list from the thread
  messages = openai_threads.messages.list(thread_id=thread.id)
  # Get the last message for the current run
  last_message = [message for message in messages.data if message.run_id == run.id and message.role == "assistant"][-1]
  # If an assistant message is found, print it
  if last_message:
    print(f"[Math Tutor]: {last_message.content[0].text.value}")
  else:
    print(f"[Math Tutor]: I'm sorry, I am not sure how to answer that. Can you ask another question?")

# Call the main function
if __name__ == "__main__":
  main()