import os
import openai
from openai_client import OpenAIClient 

# Set up OpenAI API credentials

client = OpenAIClient() 
openai.api_key = client.api_key 

def process_file_contents(file_contents: str):
    goals = """Develop a treatment plan for the patient below using this format:
        1. Specific tissue diagnosis and stage
        2. Goals of treatment
        3. Initial treatment plan and proposed duration
        4. Expected common and rare toxicities during treatment and their management
        5. Expected long-term effects of treatment
        6. Psychosocial and supportive care plans
        7. Advanced care directives and preferences
        """

    messages = [
        {"role": "user", "content": goals},
        {"role": "user", "content": file_contents}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=700,
        temperature=0.75,
        messages=messages
    )
    return response.choices[0].message.content


# Directory path where the text files are located
# e.g. directory = "/Users/carolineserapio/Desktop/stemfellowship"
directory = os.getcwd() 

# Output file path to write the results
# e.g. output_file = "/Users/carolineserapio/Desktop/stemfellowship/output.txt"
output_file = os.path.join(directory, "output.txt")

# Open the output file in write mode
with open(output_file, "w") as output:
    # Iterate over the text files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and filename != 'output.txt':
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                file_contents = file.read()
                decision_support = process_file_contents(file_contents)
                output.write(f"File: {filename}\n")
                output.write(f"Decision Support:\n{decision_support}\n\n")
                print(f"Processed file: {filename}")

print("All files processed. Results written to the output file.")