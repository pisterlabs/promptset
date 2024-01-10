import os
import csv
from src.utils.openai_api import OpenAI_API

# Initialize OpenAI API
api = OpenAI_API()

# Read prompts from CSV file
with open("tests/data/prompts.csv", "r") as file:
    reader = csv.DictReader(file)
    prompts = list(reader)

# Read customer requests from CSV file
with open("tests/data/requests.csv", "r") as file:
    reader = csv.DictReader(file)
    requests = list(reader)

# Get the directory in which this script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Open results CSV file for writing
with open(
    os.path.join(script_dir, "results", "results.csv"), "w", newline=""
) as file:
    writer = csv.DictWriter(
        file, fieldnames=["prompt_id", "request_id", "response", "cost"]
    )
    writer.writeheader()

    # For each combination of prompt and customer request...
    for prompt in prompts:
        for request in requests:
            # Make API call
            response = api.call_llm(prompt["text"], request["text"])

            # Write result to CSV file
            writer.writerow(
                {
                    "prompt_id": prompt["id"],
                    "request_id": request["id"],
                    "response": response["choices"][0]["message"][
                        "content"
                    ],  # Assuming you want the first response message
                    "cost": api.total_cost,  # Assuming total_cost is a public attribute
                }
            )
