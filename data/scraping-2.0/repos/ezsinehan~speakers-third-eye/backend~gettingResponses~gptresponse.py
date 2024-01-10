import openai
import json

# Set your OpenAI API key here
openai.api_key = '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'

# Read the dataset from the JSON file
with open('/Users/sinehanezhilmuthu/Desktop/csShit/stev2/backend/gettingData/parsingPredictions/parsed_data.json', 'r') as file:
    dataset = json.load(file)

# Convert the dataset to a string
dataset_str = json.dumps(dataset)

# Create a GPT-3 prompt to generate recommendations based on the dataset
prompt = f"Based on the provided dataset:\n{dataset_str}\n\nPlease provide specific recommendations related to the audiences emotions to the speaker whilist citing to parts in their speech using quotes, I want at least one quote and give a better example on how they could rephrase their speech, on how to increase positive emotions like interest and reduce negative emotions like confusion. Note that the text in the dataset is the speaker's speech. Only one reccomendation per run is needed, but go in detail."

# Generate a response from GPT-3
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=500
)

# Extract the generated recommendations from the response
recommendations = response.choices[0].text

# Convert the recommendations to a dictionary
recommendations_dict = {"recommendations": recommendations}

# Save the recommendations as a JSON file
with open('recommendations.json', 'w') as output_file:
    json.dump(recommendations_dict, output_file)
