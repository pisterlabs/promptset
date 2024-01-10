from openai import OpenAI
client = OpenAI()

my_updated_assistant = client.beta.assistants.update(
  "asst_abc123",
  instructions="You are an HR bot, and you have access to files to answer employee questions about company policies. Always response with info from either of the files.",
  name="HR Helper",
  tools=[{"type": "retrieval"}],
  model="gpt-4",
  file_ids=["file-abc123", "file-abc456"],
)

print(my_updated_assistant)



"""Modify assistantBeta

post https://api.openai.com/v1/assistants/{assistant_id}

Modifies an assistant.
Path parameters
assistant_id
string
Required

The ID of the assistant to modify.
Request body
model
Optional

ID of the model to use. You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.
name
string or null
Optional

The name of the assistant. The maximum length is 256 characters.
description
string or null
Optional

The description of the assistant. The maximum length is 512 characters.
instructions
string or null
Optional

The system instructions that the assistant uses. The maximum length is 32768 characters.
tools
array
Optional
Defaults to []

A list of tool enabled on the assistant. There can be a maximum of 128 tools per assistant. Tools can be of types code_interpreter, retrieval, or function.
file_ids
array
Optional
Defaults to []

A list of File IDs attached to this assistant. There can be a maximum of 20 files attached to the assistant. Files are ordered by their creation date in ascending order. If a file was previosuly attached to the list but does not show up in the list, it will be deleted from the assistant.
metadata
map
Optional

Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long.
Returns

The modified assistant object."""



"""{
  "id": "asst_abc123",
  "object": "assistant",
  "created_at": 1699009709,
  "name": "HR Helper",
  "description": null,
  "model": "gpt-4",
  "instructions": "You are an HR bot, and you have access to files to answer employee questions about company policies. Always response with info from either of the files.",
  "tools": [
    {
      "type": "retrieval"
    }
  ],
  "file_ids": [
    "file-abc123",
    "file-abc456"
  ],
  "metadata": {}
}
"""