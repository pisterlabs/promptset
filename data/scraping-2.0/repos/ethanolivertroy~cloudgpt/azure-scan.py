import os
import csv
import random
import argparse
import re
from datetime import datetime
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
import openai
from core.policy import Policy

parser = argparse.ArgumentParser(description='Retrieve all Azure policies and check for vulnerabilities')
parser.add_argument('--key', type=str, required=True, help='OpenAI API key')
parser.add_argument('--subscription-id', type=str, required=True, help='Azure subscription ID')
parser.add_argument('--redact', action='store_true', default=True, help='Redact sensitive information in the policy document (default: True)')

results = []
openai.api_key = ''

def redact_policy(policy):
    new_policy = policy
    new_policy.original_document = str(policy.policy)

    # Replace sensitive information with random values
    match = re.search(r'\b\d{12}\b', new_policy.original_document)
    if match:
        original_account = match.group()
        new_account = random.randint(100000000000, 999999999999)
        new_policy.map_accounts(original_account, new_account)
        new_policy.redacted_document = new_policy.original_document.replace(original_account, str(new_account))
    else:
        new_policy.redacted_document = new_policy.original_document

    return new_policy

def check_policy(policy):
    prompt = f'Does this Azure policy have any security vulnerabilities: \n{policy.redacted_document}'
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stream=False,
    )
    policy.ai_response = response.choices[0]['text'].strip()
    is_vulnerable = policy.is_vulnerable()
    log(f'Policy {policy.name} [{is_vulnerable}]')

    return policy

def preserve(filename, results):
    header = ['subscription_id', 'resource_group', 'name', 'id', 'vulnerable', 'policy', 'mappings']
    mode = 'a' if os.path.exists(filename) else 'w'

    log(f'Saving scan: {filename}')

    with open(filename, mode) as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if mode == 'w':
            writer.writeheader()
        for data in results:
            mappings = '' if len(data.retrieve_mappings()) == 0 else data.retrieve_mappings()
            row = {
                'subscription_id': data.subscription_id, 'resource_group': data.resource_group, 'name': data.name, 
                'id': data.id, 'vulnerable': data.ai_response, 'policy': 
                data.original_document, 'mappings': mappings
            }
            writer.writerow(row)

def log(data):
    print(f'[*] {data}')

def main(args):
    openai.api_key = args.key

    credential = DefaultAzureCredential()
    resource_client = ResourceManagementClient(credential, args.subscription_id)

    scan_utc = datetime.utcnow().strftime("%Y-%m-%d-%H%MZ")

    log(f'Retrieving and redacting policies for subscription: {args.subscription_id}')

    # Iterate over all policies in all resource groups
    for group in resource_client.resource_groups.list():
       
