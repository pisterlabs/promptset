import csv
import openai
import json
import requests

# Function to classify treatments
def classify_treatments(treatments, api_key):
    data = {
        "model": "gpt-4",  # Ensure this is the correct model identifier
        "messages": [
            {
                "role": "user",
                "content": (
                    "Here is a list of medical treatments for ARDS. Please classify each treatment into one of the ARDS treatment actions and return the classifications in JSON format use double quotes instead of single quotes. The ARDS treatment actions are:\n"
                    "Action 0 - Basic Supportive Care: Maintain basic physiological functions and patient comfort.\n"
                    "Action 1 - Advanced Respiratory Support: Enhance oxygenation and support breathing.\n"
                    "Action 2 - Metabolic and Electrolyte Management: Maintain metabolic homeostasis and correct electrolyte imbalances.\n"
                    "Action 3 - Treatment of Comorbid Conditions and Complications: Address additional health issues that could impact ARDS recovery.\n"
                    "Example: {\"Aetaminophen\": \"0\", \"Mechanical Ventilation\": \"1\", \"Potassium Chloride\": \"2\", \"Antivirals\": \"3\"}\n"
                    "Now classify the following treatments: " + ', '.join(treatments)
                )
            }
        ],
        "temperature": 0.5,
        "max_tokens": 4000
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    response_data = response.json()
    print(response_data)

    if 'error' in response_data:
        return "Error: " + response_data['error']['message']
    else:
        # Extracting the content from the last message in the response
        last_message = response_data['choices'][0]['message']['content']
        return last_message.strip()

# Main script
def main():
    api_key = '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'  # Replace with your OpenAI API key
    openai.api_key = api_key

    input_file = 'ARDS_treatments.csv'  # Replace with your CSV file path
    output_file = 'classification_results.json'

    try:
        with open(input_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            treatments = [row['drug'] for row in reader]
        

        # Split the list into chunks of 100 treatments each
        chunks = [treatments[i:i + 50] for i in range(0, len(treatments), 50)]
        
        results = {}
        for chunk in chunks:
            classification = classify_treatments(chunk, api_key)
            if classification:
                # Assuming the classification returns a JSON string
                chunk_results = json.loads(classification)
                results.update(chunk_results)

        print(results)
        # Write the results to a JSON file
        with open(output_file, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=4)

        print("Classification completed. Check the output file for results.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
