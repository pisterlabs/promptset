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

FILE_RANGE = [0, 800]

schema = {
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "The name of the firm."
    },
    "location": {
      "type": "string",
      "description": "The location of the firm."
    },
    "business": {
      "type": "string",
      "description": "The primary line of business of the firm."
    },
    "products": {
      "type": "string",
      "description": "Description of products manufactured or services offered."
    },
    "contact": {
      "type": "object",
      "properties": {
        "office": {
          "type": "string",
          "description": "Office address."
        },
        "phone": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "List of phone numbers."
        },
        "postgiro": {
          "type": "string",
          "description": "Postgiro number for the firm."
        },
        "head_office": {
          "type": "string",
          "description": "Head office address."
        },
        "sales_offices": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string"
              },
              "phone": {
                "type": "string"
              }
            }
          },
          "description": "List of sales office locations and their contact numbers."
        },
        "telegraph": {
          "type": "string",
          "description": "Telegraph address."
        },
        "factory_office": {
          "type": "string",
          "description": "Factory office address."
        },
        "factory_phone": {
          "type": "string",
          "description": "Factory phone number."
        }
      },
      "description": "Contact information for the firm."
    },
    "financials": {
      "type": "object",
      "properties": {
        "capital": {
          "type": "string",
          "description": "The capital of the firm."
        },
        "employees": {
          "type": "object",
          "additionalProperties": {
            "type": "integer"
          },
          "description": "Number of employees in different departments."
        },
        "annual_turnover": {
          "type": "string",
          "description": "Annual turnover of the firm."
        },
        "annual_production_value": {
          "type": "string",
          "description": "Annual production value."
        },
      },
      "description": "Financial information of the firm."
    },
    "foundation_year": {
      "type": "integer",
      "description": "The year the firm was founded."
    },
    "current_name_year": {
      "type": "integer",
      "description": "The year the firm's current name was adopted, if applicable."
    },
    "predecessor": {
      "type": "string",
      "description": "Predecessor firm, if any."
    },
    "original_name": {
      "type": "string",
      "description": "Original name of the firm, if it has changed."
    },
    "corporation_year": {
      "type": "integer",
      "description": "The year the firm was incorporated, if applicable."
    },
    "board": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          },
          "role": {
            "type": "string"
          }
        }
      },
      "description": "List of board members and their roles."
    },
    "owners": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of owners of the firm."
    },
    "director": {
      "type": "string",
      "description": "Name of the director."
    },
    "managing_director": {
      "type": "string",
      "description": "Name of the managing director."
    },
    "sales_manager": {
      "type": "string",
      "description": "Name of the sales manager."
    },
    "manager": {
      "type": "string",
      "description": "Name of the manager."
    },
  },
  "required": ["name", "location", "business"]
}

def structure_company_info(page_text):
    try:
        # Translate the Swedish text to English
        structure_prompt = f"Task: read the schema and return RFC compliant JSON information about the Swedish firms that is provided below:\nHere is the schema: {schema}.\nHere is the text: {page_text}.\nUse a numeric index for each firm in your JSON output and return information about all of them. Keep the business descriptions in Swedish. If there is no information for a key, leave it out."
        structure_response = client.chat.completions.create(model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are an expert on Swedish firms and business directories."},
            {"role": "user", "content": f"{structure_prompt}"}
        ])

        structured_firm_info = json.loads(structure_response.choices[0].message.content)

        return structured_firm_info

    except Exception as e:
        print(f"Error in structure_company_info: {e}")
        return None


def main():
    input_directory = Path(data_dir) / "raw/svindkal"
    output_directory = Path(data_dir) / "processed/svindkal"
    os.makedirs(output_directory, exist_ok=True)

    # Get the list of all .txt files in the input directory
    all_files = sorted([f for f in os.listdir(input_directory) if f.endswith(".txt")])

    # Process a range of files
    for file_name in all_files[FILE_RANGE[0]:FILE_RANGE[1]]:
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
                data = {
                    "original": page_text_in,
                    "structured": structured_firm_info
                }

                # Save the JSON data to the output directory
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    json.dump(data, output_file, ensure_ascii=False, indent=4)

                print(f"Processed file: {file_name}", flush=True)
            else:
                print(f"Error processing file {file_name}: translation or structuring failed. Check the API response for more information.")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

if __name__ == "__main__":
    main()