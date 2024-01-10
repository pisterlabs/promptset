import os
import requests

folder = os.path.join('/Users/joyce/desktop/544_project/', 'original_abstract')
os.makedirs(folder)
for i in range(len(data)):
    file = os.path.join(folder, 'abstract' + str(i) + '.txt')
    with open(file, 'w') as f:
        f.write(''.join(data[i]['abstract_text']))


# Function to call the OpenAI API and get an extractive summary
def get_extractive_summary(text):
    api_key = ""  # Replace with your actual OpenAI API key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": "Provide an extractive summary of the following text: " + text,
        "temperature": 0,
        "max_tokens": 150,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    response = requests.post(
        "https://api.openai.com/v1/engines/text-davinci-003/completions",
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print("Here's the response content:")
        print(response.text)
        return None

    try:
        response_json = response.json()
        summary = response_json["choices"][0]["text"].strip()
        return summary
    except KeyError:
        print("KeyError: 'choices' not found in the response. Full response:")
        print(response.text)
        return None


# Specify the directory where your files are located
folder_path = '/Users/joyce/desktop/544_project/original_abstract'

folder = os.path.join('/Users/joyce/desktop/544_project/', 'summarized_abstract')
os.makedirs(folder)

# Loop through each file in the directory
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # Check if the file is a text file
        file_path = os.path.join(folder_path, filename)

        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()

        # Get the extractive summary from OpenAI API
        summary = get_extractive_summary(content)

        # Check if summary was successfully received
        if summary:
            # Write the summary to a new file
            with open(os.path.join(folder, filename), 'a') as output_file:
                output_file.write(summary)

print("All summaries have been written to the output file.")
