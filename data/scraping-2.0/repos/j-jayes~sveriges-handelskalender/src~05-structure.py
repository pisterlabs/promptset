import os
from openai import OpenAI
import json
import yaml
from pathlib import Path
import dotenv

# Get the current working directory (root directory of the project)
root_dir = Path.cwd()
data_dir = root_dir / "data"
dotenv.load_dotenv()

# Check if script is running within GitHub Actions
if os.environ.get("GITHUB_ACTIONS") == "true":
    openai_api_key = os.environ.get("OPEN_AI_KEY")
    if not openai_api_key:
        raise ValueError("No OpenAI API key found in environment variables!")
else:
    # Access the value of OPEN_AI_KEY from the environment variables
    openai_api_key = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=openai_api_key)

FILE_RANGE = [200, 434]

schema = {
    "type": "object",
    "required": ["company_name", "activity", "establishment_year"],
    "properties": {
        "company_name": {"type": "string", "description": "The name of the company."},
        "pnr": {
            "type": "string",
            "description": "The company's identification number.",
        },
        "address": {"type": "string", "description": "The address of the company."},
        "phone_number": {
            "type": "string",
            "description": "The company's phone number.",
        },
        "bank_details": {
            "type": "object",
            "properties": {
                "bankgiro": {"type": "string"},
                "postgiro": {"type": "string"},
            },
        },
        "activity": {
            "type": "string",
            "description": "Description of the company's main business activity.",
        },
        "employees": {
            "type": "integer",
            "description": "Number of employees in the company.",
        },
        "annual_turnover": {
            "type": "number",
            "description": "Annual turnover of the company.",
        },
        "annual_production_value": {
            "type": "string",
            "description": "Annual production value.",
        },
        "capital": {"type": "number", "description": "Capital of the company."},
        "key_personnel": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "position": {"type": "string"},
                },
            },
        },
        "establishment_year": {
            "type": "integer",
            "description": "The year of establishment of the company.",
        },
    },
}


def structure_company_info(page_text):
    try:
        # Translate the Swedish text to English
        structure_prompt = f"Task: read the schema and return RFC compliant JSON information about the Swedish firms that is provided below:\nHere is the schema: {schema}.\nHere is the text: {page_text}.\nUse a numeric index for each firm in your JSON output and return information about all of them. Keep the business descriptions in Swedish. If there is no information for a key, leave it out."
        structure_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert on Swedish firms and business directories.",
                },
                {"role": "user", "content": f"{structure_prompt}"},
            ],
        )

        structured_firm_info = json.loads(structure_response.choices[0].message.content)

        return structured_firm_info

    except Exception as e:
        print(f"Error in structure_company_info: {e}")
        return None


def main():
    input_directory = Path(data_dir) / "processed/handkal_batches"
    output_directory = Path(data_dir) / "processed/handkal_structured"
    os.makedirs(output_directory, exist_ok=True)

    # Get the list of all .txt files in the input directory
    all_files = sorted([f for f in os.listdir(input_directory) if f.endswith(".txt")])

    # Process a range of files
    for file_name in all_files[FILE_RANGE[0] : FILE_RANGE[1]]:
        file_path = os.path.join(input_directory, file_name)

        try:
            # Check if the output file already exists
            output_file_name = os.path.basename(file_path).replace(".txt", ".json")
            output_file_path = os.path.join(output_directory, output_file_name)

            if os.path.exists(output_file_path):
                print(f"Skipping file {file_name}: output file already exists.")
                continue

            # Read the original Swedish company information
            with open(file_path, "r", encoding="utf-8") as file:
                page_text_in = file.read()

            # Translate the biography to English and structure it
            structured_firm_info = structure_company_info(page_text_in)

            if page_text_in is not None and structured_firm_info is not None:
                # Prepare JSON data
                data = {"original": page_text_in, "structured": structured_firm_info}

                # Save the JSON data to the output directory
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    json.dump(data, output_file, ensure_ascii=False, indent=4)

                print(f"Processed file: {file_name}", flush=True)
            else:
                print(
                    f"Error processing file {file_name}: translation or structuring failed. Check the API response for more information."
                )

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")


if __name__ == "__main__":
    main()
