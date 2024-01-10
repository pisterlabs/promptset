import time
from openai import OpenAI
from tqdm import tqdm

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI()

# watch your cost!
# https://openai.com/pricing
# Upload a file with an "assistants" purpose
# file = client.files.create(
#   file=open("webinar.txt", "rb"),
#   purpose='assistants'
# )

# You can also upload new files using curl:
# curl https://api.openai.com/v1/files \
# -H "Authorization: Bearer $OPENAI_API_KEY" \
# -F purpose="assistants" \
# -F file="@/PATH/TO/FILE.txt"

# List all files uploaded to your account
# curl https://api.openai.com/v1/files \    
#  -H "Authorization: Bearer $OPENAI_API_KEY" 


# "file-iDAtEXEDBMlSfuYRgY4F916o"  agentsinproduction.txt
# "file-x5XNjYfQnmJEpSmiJb9tQPPn" webinar.txt
# "file-fR8j7knjypVPO6AhpfdEnO9b" Vídeo 1 El shock del futuro
# "file-RBySQWib5fDVYsibN4Mg8RdO"   (video 1 Future shock)
# "file-cIZrEfjlxPAkcH7uLuJCqeLH" Vídeo 2 Los puntos sobre las IAs
# "file-YVBJfz7M7rjRIx6TDpd8rIDx", (video 2 The points on AI (costs and risks))
# 
# print(file.id)

# Add the file to the assistant
assistant = client.beta.assistants.create(
  instructions="You are a teacher assistant chatbot with access to the transcripts of the main teacher webinars. Use your knowledge base to best respond to students' queries.",
  model="gpt-4-1106-preview",
  tools=[{"type": "retrieval"}],
  file_ids=["file-RBySQWib5fDVYsibN4Mg8RdO", "file-YVBJfz7M7rjRIx6TDpd8rIDx"]
)
thread = client.beta.threads.create()

while True:
  # Get input
  prompt = input("Your question: ")

  message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=prompt,
  #  file_ids=[file.id]
  )

  run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Translate the user's question to English, if needed. When answering, translate your answer to Spanish."
  )

  # This creates a Run in a queued status. 
  run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
  )

  pbar = tqdm(total=5)
  pbar.set_description("Retrieving answer...")
  
  # periodically retrieve the Run to check on its status to see if it has moved to completed.
  while run.status != "completed":
    # print(run.status)
    run = client.beta.threads.runs.retrieve(
      thread_id=thread.id,
      run_id=run.id
    )
    # sleep for 3 seconds
    time.sleep(3)
    pbar.set_description(f"Current state: {run.status}")
    pbar.update(1)

  pbar.close()

  messages = client.beta.threads.messages.list(
    thread_id=thread.id
  )

  message = messages.data[0]
  print(message.content[0].text.value)
  print(message.content[0].text.annotations)

  # it works, but it's getting empty annotations
  # See : https://community.openai.com/t/assistant-citations-annotations-array-is-always-empty/476752
    