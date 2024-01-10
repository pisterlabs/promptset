# Purpose of this test:  Test tiktoken encoding and decoding and counting from the command line

# this code was inspoired by the openai cookbook:
  # How to count tokens with tiktoken
  # source:  https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

import tiktoken
import time
import openai

encoding = tiktoken.get_encoding("cl100k_base")  # Using the cl100k_base model.
#encoding = tiktoken.get_encoding("gpt-4-0314")  # Using the gpt-4-0314 model.

strInput = input("This code will insert a string into a chat and count the tokens.\nEnter a string:\n")

model="gpt-4-0314"  #cl100k_base

example_messages = [
    {
        "role": "system",
        "content":  "You are a helpful assistant."
    },
    {
        "role": "user", 
        "content": strInput
    }
]

# t = 9019
# print("  tiktoken token: " + str(t) + " | tiktoken string: " + encoding.decode([t]) + " | tiktoken byte literal: " + str(encoding.decode_bytes([t])))

time.sleep(1)
print("\nCount tokens using", model, "model for this example chat:\n", example_messages, "\n"); time.sleep(1)

print("Looping through chat...\n"); time.sleep(1)
def num_tokens_from_messages(messages):
    tokens_per_message = 3  #gpt-4-0314
    tokens_per_name = 1  #gpt-4-0314
    num_tokens = 0
    print("Chat array has " + str(len(messages)) + " messages, i.e. json objects" )
    num_tokens += 3  #gpt-4-0314
    # every reply is primed with <|start|>assistant<|message|>
    print("+ 3 tokens per chat array to prime the reply with special tokens <|im_start|>assistant<|im_sep|> (gpt-4)"); time.sleep(1)
    for message in messages:
        print("\nProcessing a message in the chat array:")
        num_tokens += tokens_per_message
        print("+", tokens_per_message, " tokens per message (json object) in the array"); time.sleep(1)
        for key, value in message.items():
            print("+ " + str(len(encoding.encode(value))) + " tokens for json key/value pair: " + key + " | " + value + " " + str(encoding.encode(value))); time.sleep(1)
            #print("string in json is encoded by tiktoken into token and then decoded into string: " + encoding.decode(encoding.encode(value)))
            for t in encoding.encode(value):  #for each token in the token array
                print("  tiktoken token: " + str(t) + " | tiktoken string: " + encoding.decode([t]) + " | tiktoken byte literal: " + str(encoding.decode_bytes([t])))
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
                print("+ 1 token per name:  " + value); time.sleep(1)
    return num_tokens

print(f"\nTotal of {num_tokens_from_messages(example_messages)} prompt tokens were counted by tiktoken.\n")
response = openai.ChatCompletion.create(
    model=model,
    messages=example_messages,
    temperature=0,
    max_tokens=1  # we're only counting input tokens here, so let's not waste tokens on the output
)


print(f"Total of {response['usage']['prompt_tokens']} prompt tokens were counted by the OpenAI API.")
print()

### Notes...
# Another app for token counting is this vercel app:  https://tiktokenizer.vercel.app
# Test tokens:  string "版本" and integers 0, 32, 16508, 22656, 33334, 33406

### Counting tokens with OpenAI functions
# openai-cookbook commit by altryne: https://github.com/openai/openai-cookbook/blob/f85ea304bb1aeb040b0da98ab9a2059955213828/examples/How_to_count_tokens_with_tiktoken.ipynb
# Suggestion from Twitter thread:  In my experience you just take the json schema, minify it and then throw it in a tokenizer. Haven’t tried with a lot of different prompts but all the ones I tried were in the same ballpark as this plus/minus 5 tokens.  https://twitter.com/simonw/status/1683682288251568131
