from openai import OpenAI
client = OpenAI()

client.embeddings.create(
  model="text-embedding-ada-002",
  input="The food was delicious and the waiter...",
  encoding_format="float"
)


"""Create embeddings

post https://api.openai.com/v1/embeddings

Creates an embedding vector representing the input text.
Request body
input
string or array
Required

Input text to embed, encoded as a string or array of tokens. To embed multiple inputs in a single request, pass an array of strings or array of token arrays. The input must not exceed the max input tokens for the model (8192 tokens for text-embedding-ada-002), cannot be an empty string, and any array must be 2048 dimensions or less. Example Python code for counting tokens.
model
string
Required

ID of the model to use. You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.
encoding_format
string
Optional
Defaults to float

The format to return the embeddings in. Can be either float or base64.
user
string
Optional

A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Learn more.
Returns

A list of embedding objects."""



"""{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        0.0023064255,
        -0.009327292,
        .... (1536 floats total for ada-002)
        -0.0028842222,
      ],
      "index": 0
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}"""
