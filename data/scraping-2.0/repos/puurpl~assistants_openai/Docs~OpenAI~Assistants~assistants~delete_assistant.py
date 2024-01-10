from openai import OpenAI
client = OpenAI()

response = client.beta.assistants.delete("asst_abc123")
print(response)



"""Delete assistantBeta

delete https://api.openai.com/v1/assistants/{assistant_id}

Delete an assistant.
Path parameters
assistant_id
string
Required

The ID of the assistant to delete.
Returns

Deletion status"""



"""{
  "id": "asst_abc123",
  "object": "assistant.deleted",
  "deleted": true
}
"""