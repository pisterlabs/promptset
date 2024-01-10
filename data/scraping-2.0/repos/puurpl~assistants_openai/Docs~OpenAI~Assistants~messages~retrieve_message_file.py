from openai import OpenAI
client = OpenAI()

message_files = client.beta.threads.messages.files.retrieve(
    thread_id="thread_abc123",
    message_id="msg_abc123",
    file_id="file-abc123"
)
print(message_files)



"""Retrieve message fileBeta

get https://api.openai.com/v1/threads/{thread_id}/messages/{message_id}/files/{file_id}

Retrieves a message file.
Path parameters
thread_id
string
Required

The ID of the thread to which the message and File belong.
message_id
string
Required

The ID of the message the file belongs to.
file_id
string
Required

The ID of the file being retrieved.
Returns

The message file object."""



"""{
  "id": "file-abc123",
  "object": "thread.message.file",
  "created_at": 1699061776,
  "message_id": "msg_abc123"
}
"""