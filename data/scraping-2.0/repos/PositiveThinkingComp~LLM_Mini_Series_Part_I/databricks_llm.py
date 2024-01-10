import click
import logging
import sys
import yaml
import os
from langchain.llms import AzureOpenAI, OpenAI
from app.api_funcs import get_job_infos, get_run, get_model, \
    trans_model, batch_mod_permission, prepare_api_docs
from pathlib import Path
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

PATH = Path(os.path.abspath(os.path.dirname(__file__)))

# Open the YAML file
conf_path = PATH / "app" / 'llm.yaml'
with open(conf_path) as f:
    config = yaml.safe_load(f)

# Use AzureOpenAI, if config contains deployment name, otherwise OpenAI
if config['model'].get('deployment_name', False):
    llm = AzureOpenAI(**config['model'])
else:
    llm = OpenAI(**config['model'])


headers = {"Authorization": f"Bearer {os.getenv('DBR_BEARER_TOKEN')}"}
updated_api_docs = prepare_api_docs()


def comma_list(comma_str: str):
    return comma_str.split(',')

def determine_api_text(updated_api_docs: dict, query: str):
    pick_api_prompt = """Please return the file name from the list {api_docs}
                        that best corresponds to the following query: {query}. \
                        DO NOT EXPLAIN your answer!
                        """
    api_docs = os.listdir(PATH / "app" / "dbr_api_docs")
    selected_api_doc = llm(pick_api_prompt.format(api_docs=api_docs, query=query)).lstrip().rstrip()
    logger.info(f"\nSelecting the following api document: {selected_api_doc}")
    api_text = updated_api_docs[selected_api_doc]
    return api_text, selected_api_doc    

# Add subcommands for commands
@click.group()
def cli():
    pass

@cli.group(help='Run machine learning model.')
def ml():
    pass

# Add commands for specific subcommands of 'ml'
@ml.command(help='Get information about a model.')
@click.argument('query', type=str)
def get_model_info(query):
    # Instruction to get model infos
    api_text, _ = determine_api_text(updated_api_docs, query)
    logger.info(get_model(llm, query, api_text, headers))

@ml.command(help='Get information about a model run.')
@click.argument('run_id', type=str)
@click.argument('query', type=str)
def get_run_info(query, run_id):
    # ID of the model run for which you'd like information.
    # Which information should be pulled from the run?
    api_text, _ = determine_api_text(updated_api_docs, query)
    logger.info(get_run(llm, run_id, query, api_text, headers))

@ml.command(help='Transition a model from one state to another.')
@click.argument('query', type=str)
def transition_model(query):
    # Instruction to transition a model.
    api_text, _ = determine_api_text(updated_api_docs, query)
    trans_model(llm, query, api_text, headers)

@cli.command(help='View job history.')
@click.argument('query', type=str)
def jobs(query):
    if ";" not in query:
        query = query + ";"
    query, response_query = query.split(";")
    api_text, _ = determine_api_text(updated_api_docs, query)
    # The query for the LLM + an optional query for the API response
    logger.info(get_job_infos(llm, query, response_query, api_text, headers))

@cli.command(help='Manage user permissions.')
@click.argument('query', type=str)
@click.argument('jobs', type=comma_list)
def permissions(jobs, query):
    api_text, api_name = determine_api_text(updated_api_docs, query)
    # Add/Get user permissions.
    batch_mod_permission(
        logger, llm, updated_api_docs, api_text, api_name, headers,
        query, jobs=jobs
    )

if __name__ == '__main__':
    cli()
