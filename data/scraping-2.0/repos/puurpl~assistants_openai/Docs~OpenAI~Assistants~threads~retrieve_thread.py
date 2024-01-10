from openai import OpenAI
client = OpenAI()

my_thread = client.beta.threads.retrieve("thread_abc123")
print(my_thread)



"""Retrieve threadBeta

get https://api.openai.com/v1/threads/{thread_id}

Retrieves a thread.
Path parameters
thread_id
string
Required

The ID of the thread to retrieve.
Returns

The thread object matching the specified ID."""



"""{
  "id": "thread_abc123",
  "object": "thread",
  "created_at": 1699014083,
  "metadata": {}
}
"""