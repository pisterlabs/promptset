from openai import OpenAI
client = OpenAI()



# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )

# print(completion.choices[0].message)


# assistant definition
assitant_role_definition =(f"You are an expert podcast host focused on highly technical content contained in academic papers. \n",
                           f"Given an academic paper, analyze and understand the paper. Carefully identify the key concepts and findinds in the paper. \n"
                           f"Finally, summarize the paper content into a 10 minutes tech podcast in which you walk the listener through the key concepts and findins in the paper.")


podcast_host = client.beta.assistants.create(
    name="Podcast Host",
    instructions=assitant_role_definition,
    tools=[{"type": "retrieval"}],
    model="gpt-4-1106-preview"
)

# assistant thread
podcast_thread = client.beta.threads.create()

# pdf_file loading
input_pdf = client.files.create(
  file=open("data/attention_paper.pdf", "rb"),
  purpose='message'
)


message = client.beta.threads.messages.create(
  thread_id=podcast_thread.id,
  role="user",
  content="Please create a new podcast based on inputted pdf.",
  file_ids=[podcast_thread.id]
)


run = client.beta.threads.runs.create(
  thread_id=podcast_thread.id,
  assistant_id=podcast_host.id
)

messages = client.beta.threads.messages.list(
  thread_id=podcast_thread.id
)

