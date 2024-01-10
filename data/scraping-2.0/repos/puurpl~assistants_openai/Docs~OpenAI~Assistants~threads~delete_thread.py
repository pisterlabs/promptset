from openai import OpenAI
client = OpenAI()

response = client.beta.threads.delete("thread_abc123")
print(response)



"""Delete threadBeta

delete https://api.openai.com/v1/threads/{thread_id}

Delete a thread.
Path parameters
thread_id
string
Required

The ID of the thread to delete.
Returns

Deletion status"""



"""{
  "id": "thread_abc123",
  "object": "thread.deleted",
  "deleted": true
}
"""