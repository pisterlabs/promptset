import json
import os
from collections import OrderedDict

import click
import openai
import pandas as pd
import yaml
from rich.console import Console

console = Console()
# Constants
BACKENDS = [
    "Pandas",
    "Spark",
    "SQLite",
    "PostgreSQL",
    "MySQL",
    "MSSQL",
    "Trino",
    "Redshift",
    "BigQuery",
    "Snowflake",
]


def select_backend():
    console.print("Select one of the following backends:")
    for i, backend in enumerate(BACKENDS, 1):
        console.print(f"{i}. {backend}")

    backend_choice = click.prompt(
        "Enter the number corresponding to the backend", type=int
    )
    if 1 <= backend_choice <= len(BACKENDS):
        selected_backend = BACKENDS[backend_choice - 1]
        console.print(
            f"You've selected {selected_backend} as the source of your data."
        )
        return selected_backend
    else:
        console.print(
            "Invalid choice. Please enter a number between 1 and",
            len(BACKENDS),
        )
        return None


def initialize_openai_setup():
    if not openai.api_key:
        # If not set, then try to fetch the API key from the environment.
        api_key = os.getenv("OPENAI_API_KEY")

        # If it's neither in openai.api_key nor in the environment, prompt the user for it.
        if not api_key:
            api_key = input("Please provide your OpenAI API key: ")

        # Set the API key for openai.
        openai.api_key = api_key
        if not openai.api_key:
            raise ValueError("API key not provided!")

    console.print("Welcome to the OpenAI Model Interaction Setup.")
    console.print(
        "By default, gpt-3.5-turbo model will be used with a temperature of 0."
    )
    customize_choice = click.prompt(
        "Would you like to customize the model settings? (yes/no)",
        default="no",
    )

    model_type = "gpt-3.5-turbo"
    temperature = 0.0
    max_tokens = None

    if customize_choice.lower() == "yes":
        model_type = click.prompt(
            "Enter the model type (or press Enter to use gpt-3.5-turbo):",
            default="gpt-3.5-turbo",
        )
        temperature = click.prompt(
            "Enter the temperature (or press Enter to use 0.5):",
            default=0.5,
            type=float,
        )
        max_tokens_input = click.prompt(
            "Enter the max tokens (or press Enter to skip):",
            default="",
            type=str,
        )
        if max_tokens_input:
            max_tokens = int(max_tokens_input)

    return model_type, temperature, max_tokens


def choose_expectations_source():
    console = Console()
    console.print(
        "A set of core expectations can be provided to the model to improve the accuracy of the output."
    )
    console.print("")
    console.print(
        "Feeding the model with core expectations will use more tokens. (8 columns ~2000 tokens)"
    )
    console.print(
        "Using model's base knowledge would consume 600-1000 tokens, but GPT will occassionally provide [bold red]non-existent[/bold red] expectations."
    )
    console.print("How do you want to proceed?")
    console.print("1. Feed the model with core expectations")
    console.print("2. Rely on the model's base knowledge")
    choice = click.prompt(
        "Please choose an option (1 or 2):", default=1, type=int
    )
    return choice == 1


def handle_output_customization(content_json):
    console.print(
        "You can choose the output format for the expectations suite:"
    )
    console.print("1. JSON File (default)")
    console.print("2. YAML File")
    console.print("3. Print to Console")

    output_choice = click.prompt("Enter your choice:", default=1, type=int)
    if output_choice == 1:
        with open("expectations_suite.json", "w") as file:
            json.dump(content_json, file, indent=4)
        console.print("\nExpectations suite saved to expectations_suite.json.")
    elif output_choice == 2:
        with open("expectations_suite.yaml", "w") as file:
            yaml.dump(content_json, file)
        console.print("\nExpectations suite saved to expectations_suite.yaml.")
    elif output_choice == 3:
        console.print("\nExpectations Suite:")
        console.print(json.dumps(content_json, indent=4))
    else:
        console.print("Invalid choice. Saving to JSON file by default.")
        with open("expectations_suite.json", "w") as file:
            json.dump(content_json, file, indent=4)


def append_results_to_yaml(content_json, file_path="output.yaml"):
    # Define the format for appending to the YAML file
    yaml_content = {"Validation": {"Suite Name": "my_suite", "Tests": []}}

    # Convert JSON content to the specified YAML format
    for expectation in content_json.get("expectations", []):
        test = {
            "expectation": expectation.get("expectation_type"),
            "kwargs": expectation.get("kwargs", {}),
        }
        yaml_content["Validation"]["Tests"].append(test)

    # Append to the YAML file
    with open(file_path, "a") as file:
        yaml.dump(yaml_content, file)

    console = Console()
    console.print(f"Results appended to {file_path}.")


def get_column_details():
    """Prompt the user for column details using ; as the delimiter."""
    columns = []

    click.echo("\nPlease provide column details in the following format:")
    click.echo("column;mode;datatype;description")
    click.echo("For mode: 'null' for nullable and 'req' for required.")
    click.echo("Description is optional and can include '/'.\n")

    click.echo("Provide the contracted column details. use q or exit to stop.")
    while True:
        column_detail = click.prompt("Enter column details", type=str)
        if column_detail.lower() == "exit" or column_detail.lower() == "q":
            break

        parts = column_detail.split(";")
        if len(parts) < 3:
            click.echo(
                "Invalid format. Please provide details in the format column;mode;datatype;description."
            )
            continue

        column_name = parts[0].strip()
        mode = "REQUIRED" if parts[1].strip().lower() == "req" else "NULLABLE"
        data_type = parts[2].strip()
        description = parts[3].strip() if len(parts) > 3 else ""

        columns.append(
            {
                "name": column_name,
                "description": description,
                "mode": mode,
                "type": data_type,
            }
        )

    return columns


def prune_empty_values(d):
    """
    Recursively remove keys with None or empty values from a dictionary.
    """
    if not isinstance(d, dict):
        return d
    clean_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = prune_empty_values(v)
        if v:  # This checks if the value is not None or empty
            clean_dict[k] = v
    return clean_dict


def yaml_content_from_json(content_json, suite_name=None):
    yaml_content = {"Validation": {"Tests": []}}

    if suite_name:
        yaml_content["Validation"]["Suite Name"] = suite_name

    for expectation in content_json.get("expectations", []):
        test = {
            "expectation": expectation.get("expectation_type"),
            "kwargs": expectation.get("kwargs", {}),
        }
        yaml_content["Validation"]["Tests"].append(test)

    return yaml.dump(
        yaml_content, default_flow_style=False, sort_keys=True, indent=2
    )
