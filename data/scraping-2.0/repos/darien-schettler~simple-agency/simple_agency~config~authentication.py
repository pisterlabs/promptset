import os, sys
import openai
from dotenv import load_dotenv, find_dotenv


def auth_openai(openai_api_key=None):
    if not openai_api_key:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        openai.api_key = openai_api_key

    if not openai.api_key.startswith("sk-"):
        raise ValueError("Please enter a valid OpenAI API key")



def find_and_load_dotenv():
    """ Load .env file for authentication """
    try:
        load_dotenv(find_dotenv())
    except FileNotFoundError:
        print("\n\n... Please upload your .env file to one of the directories in this project ...\n")


def return_infra_keys(infra_to_use=None):
    """ Return the infrastructure keys

    Args:
        infra_to_use (str): The infrastructure to use. Defaults to "paperspace" if none is specified.

    Returns:
        dict: The required access credentials for the specified infrastructure

    Raises:
        ValueError: If the specified infrastructure type is not supported
    """

    # Load the .env file to ensure all environment variables are available
    find_and_load_dotenv()
    auth_openai()

    # Setting up the infrastructure - default to paperspace if no infrastructure is specified or found
    if infra_to_use is None:
        infra_to_use = os.getenv("INFRASTRUCTURE", "paperspace")

    #########################
    # I think I can replace the below with RH config/auth as it takes care of all of this
    #########################

    # Cloud Credentials
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    GCP_SERVICE_ACCOUNT_FILE = os.getenv("GCP_SERVICE_ACCOUNT_FILE")
    AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
    AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
    AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
    PAPERSPACE_IP_ADDRESS = os.getenv("PAPERSPACE_IP_ADDRESS")
    PAPERSPACE_HOST_NAME = os.getenv("PAPERSPACE_HOST_NAME")

    # Mapping infrastructure to required keys
    INFRASTRUCTURE_KEYS = {
        "aws": {
            "access_key_id": AWS_ACCESS_KEY_ID,
            "secret_access_key": AWS_SECRET_ACCESS_KEY,
        },
        "gcp": {
            "project_id": GCP_PROJECT_ID,
            "service_account_file": GCP_SERVICE_ACCOUNT_FILE,
        },
        "azure": {
            "tenant_id": AZURE_TENANT_ID,
            "client_id": AZURE_CLIENT_ID,
            "client_secret": AZURE_CLIENT_SECRET,
        },
        "paperspace": {
            "ip_address": PAPERSPACE_IP_ADDRESS,
            "host_name": PAPERSPACE_HOST_NAME,
        }
    }
    #########################

    infra_access_creds = INFRASTRUCTURE_KEYS.get(infra_to_use)
    if infra_access_creds is None:
        raise ValueError(f"Infrastructure type of '{infra_to_use}' is not supported. "
                         f"Please try again with one of {list(INFRASTRUCTURE_KEYS.keys())}")
    else:
        return infra_access_creds
