import os
import glob
from dotenv import load_dotenv
from openai import OpenAI
import datetime

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def read_model_output(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def send_to_gpt(summary):
    prompt = ("You are an expert statistician trained in selecting the best model "
              "for forecasting. Please review the following regression model summary "
              "and provide an insightful analysis, highlighting key findings and suggestions. Provide your own interpretations and make bold predictions.:\n\n"
              f"{summary}")

    response = client.completions.create(
        model='text-davinci-003',  # Replace 'curie' with the model you're using
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

def get_latest_file(path, prefix):
    # List all files in the given directory with the specified prefix
    file_list = glob.glob(os.path.join(path, prefix + '*'))
    
    # Sort files by modification time in descending order
    file_list.sort(key=os.path.getmtime, reverse=True)

    # Return the first file in the list (most recent one)
    if file_list:
        return file_list[0]
    else:
        return None

# Example usage
# Split the path and the prefix
path = '../logs/csv_files/combination_regressions/combination_regressions_filtered'
# path = '../logs/csv_files/multiple_regressions/ranked'
prefix = 'top_combination_models.csv'
# Call the function with both arguments
latest_file = get_latest_file(path, prefix)
print(f"The latest file is: {latest_file}")

# Use the latest file for reading the model summary
model_summary_file = latest_file

# Read the output and send to GPT
model_summary = read_model_output(model_summary_file)
analysis = send_to_gpt(model_summary)

print(analysis)

# Get the analysis output
analysis = send_to_gpt(model_summary)

# Get the current timestamp for the filename
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Create a path for the new file
output_file_path = f"../logs/text_files/gpt_summaries/gpt_analysis_{timestamp}.txt"

# Write the analysis to the file
with open(output_file_path, 'w') as file:
    file.write(analysis)

print(f"Analysis saved to {output_file_path}")