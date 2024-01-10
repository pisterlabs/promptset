import csv
import openai
import time
import logging
import re
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables from a .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    filename='./readme_analyzer/error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Function to predict the domain of a project using OpenAI's model
def predict_domain(project_description):
    """Predict the domain using OpenAI's GPT model."""
    try:
        # Define the model name
        model_name = "gpt-3.5-turbo"

        # Truncate to 4000 characters if necessary
        truncated_description = project_description[:4000] if len(project_description) > 4000 else project_description

        # Define the chat model's parameters with a clear and concise instruction
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent assistant specialized in classifying projects into specific domains based on their descriptions. Your task is to analyze a project's README and assign an appropriate domain from the following categories: Application Software, System Software, Web Libraries and Frameworks, Non-Web Libraries and Frameworks, Software Tools, and Documentation."
            },
            {
                "role": "user",
                "content": f"Here is the project README description:\n\n{truncated_description}\n\nBased on the description provided, into which of the specified domains does this project best fit?"
            }
        ]

        # Make an API call to OpenAI
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages
        )

        # Extract and return the model's response
        domain_prediction = response['choices'][0]['message']['content'].strip()

        # Return the predicted domain
        return domain_prediction
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    # Return None if there was an exception
    return None

def remove_links_and_tags(text):
    """Remove links and markdown tags from the text."""
    return re.sub(r'\(http[^\)]*\)|!\[[^\]]*\]|\[[^\]]*\]', '', text)

def analyze_openai():
    """Analyze project readmes and predict their domains."""
    # Define the CSV file paths
    input_csv = './readme-mining/readme_content_filtered_final.csv'
    output_csv = './readme_analyzer/project_domains_classifier_final.csv'
    
    # Set OpenAI API key from environment variables
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Read and write CSV files using context managers
    with open(input_csv, 'r', newline='', encoding='utf-8') as infile, \
         open(output_csv, 'a', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Skip the header row
        next(reader)

        # Iterate over the rows in the input CSV
        for counter, row in enumerate(reader, 1):
            project_name, readme_content = row[0], remove_links_and_tags(row[1])

            # Call the function to predict the domain
            domain = predict_domain(readme_content)

            if domain:
                # Write the project_name and domain to the output CSV
                writer.writerow([project_name, domain])
                print(f"Domain predicted for {project_name}: {domain}")
            else:
                print(f"Failed to predict domain for {project_name}. See error_log.txt for details.")

            # Counter information
            print(f"Processed {counter} projects")

            # Sleep to avoid hitting API rate limits
            time.sleep(20)

if __name__ == "__main__":
    analyze_openai()
