import openai
import os

# Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
openai.api_key = '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'

def format_song_string(input_string):
    prompt = f"Reformat the string {input_string} into a CSV format. Provide only the values for id, artist, and track, separated by commas, and ensure all letters are in lowercase. Make sure to remove irrelevant words from the provided string. Respond with only the csv string. I dont want comments from you"

    response = openai.Completion.create(
        engine="text-davinci-003",  # You can change the engine as needed
        prompt=prompt,
        temperature=0,  # Set to 0 for deterministic responses
        max_tokens=150,  # Adjust as needed
        stop=None  # Can add stop words if needed
    )
    print(response)
    formatted_string = response.choices[0].text.strip()
    return formatted_string



input_folder = "../raw-data/"
file_names = os.listdir(input_folder)
file_names = sorted(file_names)[-489:]


with open('id-artist-title.csv', 'a') as file:
    file.write("id,artist,title")
    for input_string in file_names:
        # Example usage
        # input_string = "023_C M Lord - Oh Mama Your Daughter's A Woman Tonight [axy7GP9jCXw].mp3"
        formatted_result = format_song_string(input_string)
        file.write(f'{formatted_result}\n')


# print("Original:", input_string)
# print("Formatted:", formatted_result)