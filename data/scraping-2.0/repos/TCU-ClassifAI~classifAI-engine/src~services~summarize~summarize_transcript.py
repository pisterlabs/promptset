import os
import openai
from datetime import datetime

# Load .env file if it exists
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Open text files
# Specify the file path
file_path = "file.txt"  # Replace with the path to your text file

# Initialize an empty string to store the file content
file_content = ""

# Open the file in 'read' mode
try:
    with open(file_path, "r") as file:
        # Read the entire content of the file into the string
        file_content = file.read()
except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")


gpt_instructions = """You will be provided with the text of a transcript. Your goal is to summarize the text in 1-2 sentences.
 Your response should be detailed, concise, and clear. The tone should be that of a clinical diagnosis.
 Use as few words as necessary without sacrificing quality.  """

prompt = gpt_instructions + "Here are the texts:" + "\n\n" + file_content

response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    # '''Temperature is a number between 0 and 2, with a default value of 1 or 0.7
    # depending on the model you choose. The temperature is used to control the randomness
    # of the output. When you set it higher, you'll get more random outputs.
    # When you set it lower, towards 0, the values are more deterministic.'''
)

print(response)
print(response["choices"][0]["message"]["content"])


# Get current time
now = datetime.now()  # current date and time

# Set the name for the response file
names = file_path.rsplit(".", 1)
# response_file_name = names[0] + "-Response" +  + ".txt"
response_file_name = names[0] + ";" + now.strftime("%m-%d-%Y;%H:%M:%S") + ".txt"

# Create the response file if it doesn't exist
if not os.path.exists(response_file_name):
    open(response_file_name, "w").close()


with open(response_file_name, "w") as file:
    # Write some text to the file
    # file.write(gpt_instructions)
    file.write(gpt_instructions + "\n\n")
    file.write(response["choices"][0]["message"]["content"])


# Previous prompt
# gpt_instructions = """ You will be provided with texts.
#     Find emotion related words.
#     Your response should be detailed, concise, and clear.
#     The tone should be that of a clinical diagnosis.
#     Use as few words as necessary without sacrificing quality.
#     Explain why do you identify them as emotion words.
#     Here are the texts: """
