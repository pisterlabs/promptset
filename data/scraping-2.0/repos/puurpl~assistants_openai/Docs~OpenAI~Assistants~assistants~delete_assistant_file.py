from openai import OpenAI
client = OpenAI()

deleted_assistant_file = client.beta.assistants.files.delete(
    assistant_id="asst_abc123",
    file_id="file-abc123"
)
print(deleted_assistant_file)



"""Delete assistant fileBeta

delete https://api.openai.com/v1/assistants/{assistant_id}/files/{file_id}

Delete an assistant file.
Path parameters
assistant_id
string
Required

The ID of the assistant that the file belongs to.
file_id
string
Required

The ID of the file to delete.
Returns

Deletion status"""



"""{
  id: "file-abc123",
  object: "assistant.file.deleted",
  deleted: true
}
"""