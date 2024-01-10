import os
import datetime
import openai

# Set up OpenAI API key
openai.api_key = "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"

# Define the folder path containing the files
folder_path = "database"

# Define the string to search for
search_string = input("Enter the string to search for: ")

# search for all files containing the search string in their name
matching_files = []
for file_name in os.listdir(folder_path):
    if search_string in file_name:
        file_path = os.path.join(folder_path, file_name)
        matching_files.append(file_path)

# print the matching file paths
if len(matching_files) == 0:
    print(f"No files found containing '{search_string}' in their name.")
else:
    print(f"The following files were found containing '{search_string}' in their name:")
    for file_path in matching_files:
        print(file_path)

# Define the paths to the documents to compare
document1_path = matching_files[0]
document2_path = matching_files[1]

# Read in the contents of the two documents
with open(document1_path, "r") as f:
    document1 = f.read()

with open(document2_path, "r") as f:
    document2 = f.read()

# Define the prompt to be used with OpenAI's GPT-3
prompt = "Match common points between the following two documents:\n\nDocument 1:\n" + document1 + "\n\nDocument 2:\n" + document2 + "\n\nMatched points:"

# Define the parameters for the OpenAI API request
params = {
    "engine": "text-davinci-002",
    "prompt": prompt,
    "temperature": 0.7,
    "max_tokens": 2048,
    "stop": "Matched points:",
}

# Send the request to the OpenAI API
response = openai.Completion.create(**params)

# Extract the matched points from the OpenAI API response
matched_points = response.choices[0].text.strip()

# Split the matched points into a list
matched_points_list = matched_points.split("\n")

# Define the filename for the matched points text file
current_time = datetime.datetime.now().strftime(" %d.%m.%Y-%H.%M.%S")
filename = f"{search_string}{current_time}.txt"
filepath = os.path.join("mathpoints", filename)

# Write the matched points to a text file
with open(filepath, "w") as f:
    for point in matched_points_list:
        f.write(point + "\n")

# Print out the matched points
print("Matched Points:\n")
for point in matched_points_list:
    print(point)

# Print out the filepath of the saved matched points
print(f"\nThe matched points were saved to the file: {filepath}")
