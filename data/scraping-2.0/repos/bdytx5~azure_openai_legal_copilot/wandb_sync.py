import os
import openai
from wandb.integration.openai.fine_tuning import WandbLogger
from openai import AzureOpenAI
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sync Wandb with OpenAI Fine-Tune')
    parser.add_argument('--id', type=str, help='Fine Tune Job ID')
    parser.add_argument('--project', type=str, help='Project Name')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing logs (defaults to False)')
    args = parser.parse_args()

    # Retrieve OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize OpenAI with the API key
    openai.api_key = openai_api_key

    # Initialize AzureOpenAI client (modify the endpoint if necessary)
    client = AzureOpenAI(
        azure_endpoint = "YOUR_AZURE_ENDPOINT_URL",  # Replace with your Azure endpoint URL
        api_key=openai_api_key,
        api_version="2023-09-15-preview"  # Required version for fine-tuning
    )

    # Sync Wandb with the fine-tune job
    WandbLogger.sync(fine_tune_job_id=args.id, openai_client=client, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
