import os
from openai import AzureOpenAI
from tqdm import tqdm  # Import tqdm for the progress bar

def read_file_to_variable(file_paths):
    # Get the size of the file for tqdm progress bar
    for file_path in file_paths:

        file_size = os.path.getsize(file_path)
        
        # Initialize tqdm with the file size
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Reading file {file_path}") as pbar:
            # Read the content from the file
            with open(file_path, "r") as file:
                file_content = file.read()
                pbar.update(file_size)  # Update the progress bar to show completion

        return file_content

    # Example usage with six different file paths

    # Read content from each file and store in variables
    variable_a, variable_b, variable_c, variable_d, variable_e, variable_f = [
        read_file_to_variable(file_path) for file_path in file_paths
    ]

    client = AzureOpenAI(
        api_key="b8e6ac2cfda244dd848a823511255a0b",
        azure_endpoint="https://hackathonservice.openai.azure.com/",
        api_version="2023-05-15"
    )

    # Generate responses for each file and store in variables
    response_a = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You take text files and give a one sentence summary"},
            {"role": "user", "content": f"Read this file and summarize it {variable_a}"},
            {"role": "assistant", "content": "Send the summary for response_a"},
        ]
    ).choices[0].message.content

    response_b = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You take text files and give a one sentence summary"},
            {"role": "user", "content": f"Read this file and summarize it {variable_b}"},
            {"role": "assistant", "content": "Send the summary for response_b"},
        ]
    ).choices[0].message.content

    response_c = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You take text files and give a one sentence summary"},
            {"role": "user", "content": f"Read this file and summarize it {variable_c}"},
            {"role": "assistant", "content": "Send the summary for response_c"},
        ]
    ).choices[0].message.content

    response_d = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You take text files and give a one sentence summary"},
            {"role": "user", "content": f"Read this file and summarize it {variable_d}"},
            {"role": "assistant", "content": "Send the summary for response_d"},
        ]
    ).choices[0].message.content

    response_e = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You take text files and give a one sentence summary"},
            {"role": "user", "content": f"Read this file and summarize it {variable_e}"},
            {"role": "assistant", "content": "Send the summary for response_e"},
        ]
    ).choices[0].message.content

    response_f = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You take text files and give a one sentence summary"},
            {"role": "user", "content": f"Read this file and summarize it {variable_f}"},
            {"role": "assistant", "content": "Send the summary for response_f"},
        ]
    ).choices[0].message.content

    # Print or use the generated responses as needed
    print(response_a)
    print(response_b)
    print(response_c)
    print(response_d)
    print(response_e)
    print(response_f)

    type (response_f)