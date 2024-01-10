# This code is for v1 of the openai package: pypi.org/project/openai
from openai import OpenAI

# initiate client
client = OpenAI()


# TODO: start with an empty message object
# TODO: start a timestamped log file for this specific conversation
# store message(s)
messages = [
    {
      "role": "user",
      "content": "what is the cutoff for the data in this version?"
    },
]

# retrieve response
response = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=messages, #type: ignore
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

message_in_response = response.choices[0].message

# TODO: consider putting the following in a loop where you ask for input

# add the newest message to the conversation object
messages = messages + [message_in_response]


# ===
# EXAMPLE CODE FROM GPT
# use this to open and close a file with a loop, also 
# will use the same loop to send requests to gpt
# ===

# # Define the name of the file
# filename = "user_input.txt"
# # The specific string that will break the loop
# exit_string = "EXIT"

# # Open the file for writing
# with open(filename, 'w') as file:
#     while True:
#         # Get input from the user
#         user_input = input("Enter some text (or type 'EXIT' to finish): ")
#         # Check if the user wants to exit
#         if user_input == exit_string:
#             print("Exiting and closing the file.")
#             break
#         # Write the input to the file along with a newline
#         file.write(user_input + "\n")

# # No need to explicitly close the file, as the 'with' block takes care of that.

