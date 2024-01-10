# Standard library imports
import json
import logging
import os
import time
import subprocess
from typing import List, Union
import importlib.util

# Third-party imports
import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool, tool
import pandas as pd
import tldextract

# Local (or relative) imports
from .utils import (
    deactivate_aws_key_helper,
    get_dependabot_alert,
    get_user_name_from_access_key,
    is_bucket_public,
    scan_git_secrets,
    extract_content_from_url,
    incident_extractor_tool,
    generate_threat_model,
    generate_required_tools_code,
    extract_elements,
    fetch_dns_records,
    fetch_tls_certificate,
    analyze_whois,
    phishing_insights_extractor_tool,
    process_user_input_url,
    analyze_file,
    checkout_branch,
    create_branch,
    apply_fix,
    commit_changes,
    push_changes,
    open_pull_request
)
# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize LLM
llm = ChatOpenAI(temperature=0)

@tool
def check_credentials_in_repo(git_repo: str) -> str:
    """
    This function checks a given git repository for any leaked credentials.
    
    Leaked credentials in a repository can pose a significant security risk. This function uses the scan_git_secrets method
    to scan the repository and return any findings.
    
    Parameters:
    git_repo (str): The URL of the git repository to check.
    
    Returns:
    results (str): The results of the scan for leaked credentials.
    """
    results = scan_git_secrets(git_repo)
    return results

@tool
def check_git_depdency_cves(git_repo:str) -> str:
    """
    This function checks a given git repository for potential vulnerabilities due to out-of-date dependencies.
    
    Out-of-date dependencies can have known security vulnerabilities that pose a risk to the repository. This function uses
    the get_dependabot_alert method to check the repository's dependencies and return any findings.
    
    Parameters:
    git_repo (str): The URL of the git repository to check.
    
    Returns:
    results (str): The results of the check for out-of-date dependencies with security issues.
    """
    results = get_dependabot_alert(git_repo)
    return results

@tool
def invalidate_aws_key(access_key: str) -> str:
    """
    This function invalidates a given AWS access key to mitigate the security risk associated with leaked credentials.
    
    If the operation is successful, it returns 'AWS Key invalidated successfully'.
    If the user associated with the key doesn't exist (which may occur if the key is already invalidated), it returns 'No user found for the given access key'.
    
    Parameters:
    access_key (str): The AWS access key to invalidate. It should be in the format: 'AKIAxxxxxxxxxxxxx'.
    
    Returns:
    response (str): The result of the key invalidation operation.
    """
    iam_client = boto3.client('iam')
    user_name = get_user_name_from_access_key(iam_client, access_key)
    if user_name:
        response = deactivate_aws_key_helper(iam_client, user_name, access_key)
    else:
        response = "No user found for the given access key: " +  str({access_key})
    print("Response in tool: ", response)
    return response

slack_token = os.getenv("slack_api_key")

@tool
def inform_SOC(message: str) -> str:
    """
    This function sends a notification to the security operation center (SOC) when a security issue is detected.
    
    If the message is delivered successfully, it returns 'Message posted successfully'.
    If there are any issues in notifying the SOC, it returns 'Error posting message'.
    
    Parameters:
    message (str): The message to be sent to the SOC.
    
    Returns:
    response (str): The result of the message delivery operation.
    """
    client = WebClient(token=slack_token)

    # Send a message to the channel
    response = client.chat_postMessage(
    channel='#soc',
    text=message
    )

    # Check the response
    if response['ok']:
        response = print('Message posted successfully.')
    else:
        response = print(f"Error posting message: {response['error']}")
    return response

@tool
def get_public_buckets(aws_account_name: str) -> list:
    """Use this tool to check if S3 buckets are left open and unautenticated for an AWS account.
    This can be a security issue in AWS account.
    It returns the list of buckets that are open and an empty list if there are no open buckets.
    Having an AWS bucket open is serious and you should inform the SOC if I have any open buckets."""
    s3_client = boto3.client('s3')
    print("Checking buckets in account: ", aws_account_name)

    # Get list of all bucket names
    response = s3_client.list_buckets()
    all_buckets = [bucket['Name'] for bucket in response['Buckets']]

    # Check each bucket to see if it's public
    public_buckets = []
    for bucket in all_buckets:
        if is_bucket_public(s3_client, bucket):
            public_buckets.append(bucket)

    return public_buckets

@tool
def check_aws_mfa(account: str) -> list:
    """Checks AWS account for users that don't have MFA enabled on platform AWS and returns them.
    Returns an empty list if there are no users without MFA on platform AWS."""
    users_without_mfa = []
    try:
        logging.info("Checking users in AWS account: %s", account)
        client = boto3.client('iam')
        users = client.list_users()['Users']
    except (NoCredentialsError, BotoCoreError, ClientError) as error:
        logging.error("Failed to retrieve IAM users: %s", error)
        return []

    for user in users:
        try:
            client.get_login_profile(UserName=user['UserName'])  # check if user has console access
            mfa_devices = client.list_mfa_devices(UserName=user['UserName'])
            if not mfa_devices['MFADevices']:
                users_without_mfa.append(user['UserName'])
        except ClientError as error:
            # If the error message is about the user not having a login profile, skip this user
            if error.response['Error']['Code'] == 'NoSuchEntity':
                continue
            else:
                logging.error("Failed to retrieve MFA devices or console access for user %s: %s", user['UserName'], error)

    return users_without_mfa

@tool
def extract_incident_schema(incident_url: str) -> dict:
    """
    This function extracts interesting insights about an incident from a given URL.

    The insights are returned as a dictionary, which can include various details about the incident such as the type of incident, the entities involved, the impact, and any remedial actions taken.

    Parameters:
    incident_url (str): The URL from which to extract incident insights.

    Returns:
    insights (dict): A dictionary containing interesting insights about the incident.
    """
    incident_content = extract_content_from_url(incident_url)
    incident_dict = incident_extractor_tool(incident_content)
    return incident_dict

@tool
def check_cve_in_kev(cve_string: str) -> Union[List[str], str]:
    """
    This function checks if a list of CVEs is actively being exploited in the wild and is part of the KEV list.
    It prioritizes CVEs that are being exploited in the wild as they pose a significant security risk.
    The function returns a list of exploited CVEs or an empty list if none are found.

    Parameters:
    cve_string (str): A string of CVEs separated by commas.

    Returns:
    exploited_cves (Union[List[str], str]): A list of CVEs that are being exploited in the wild or an error message if an error occurs.
    """
    cisa_kev_url = "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"
    
    try:
        df = pd.read_csv(cisa_kev_url)
    except Exception as e:
        logging.error(f"Error occurred while reading CSV: {e}")
        return f"Error occurred while reading CSV: {e}"
    
    if 'cveID' not in df.columns:
        logging.error("Error: 'CVE' column not found in DataFrame")
        return "Error: 'CVE' column not found in DataFrame"

    try:
        cve_list = [cve.strip() for cve in cve_string.split(",")]
    except Exception as e:
        logging.error(f"Error occurred while splitting the CVE string: {e}")
        return f"Error occurred while splitting the CVE string: {e}"

    exploited_cves = [cve for cve in cve_list if cve in df['cveID'].values]
    logging.info(f"Exploited CVEs found: {exploited_cves}")

    return exploited_cves

@tool
def threat_model(url: str) -> str:
    """
    This function generates a threat model for a given URL.
    
    It first retrieves the content from the URL using the get_content_from_url function. 
    Then, it passes the retrieved content to the generate_threat_model function to generate the threat model.
    
    Parameters:
    url (str): The URL for which to generate the threat model.
    
    Returns:
    threat_model (str): The generated threat model.
    """
    try:
        content = extract_content_from_url(url)
    except Exception as e:
        logging.error(f"Error occurred while getting content from URL: {e}")
        return f"Error occurred while getting content from URL: {e}"

    try:
        threat_model = generate_threat_model(content)
    except Exception as e:
        logging.error(f"Error occurred while generating threat model: {e}")
        return f"Error occurred while generating threat model: {e}"

    return threat_model

@tool
def extract_phishing_insights(url: str) -> str:
    """
    This function extracts phishing insights for a given URL.
    
    It first extracts the domain and host from the URL. Then, it retrieves the rendered content, DNS record, 
    TLS record, and WHOIS records for the domain. These records are then passed to the phishing_insights_extractor_tool 
    function to generate the phishing insights.
    
    Parameters:
    url (str): The URL for which to extract phishing insights.
    
    Returns:
    phishing_insights (str): The extracted phishing insights.
    """
    try:
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"
        host = f"{extracted.subdomain}.{domain}" if extracted.subdomain else domain
        rendered_content = extract_elements(url)
        dns_record = fetch_dns_records(domain)
        tls_record = fetch_tls_certificate(host)
        who_is_records = analyze_whois(domain)
        input = url + str(dns_record) + str(tls_record) + str(who_is_records) + str(rendered_content)
        phishing_insights = phishing_insights_extractor_tool(input)
        return phishing_insights
    except Exception as e:
        logging.error(f"Error occurred while extracting phishing insights: {e}")
        return f"Error occurred while extracting phishing insights: {e}"

@tool
def analyze_and_fix_vulnerabilities(repo_path_url: str) -> str:
    """
    This function analyzes and fixes vulnerabilities in a given repository.
    
    It first processes the user input URL to get the file paths and repository URL. Then, it checks out the primary branch 
    and analyzes each file for vulnerabilities. If a vulnerability is found, it checks out the primary branch, creates a new 
    branch, applies the fix, commits the changes, and pushes the changes to the new branch. Finally, it opens a pull request 
    with the fix.
    
    Parameters:
    repo_path_url (str): The URL of the repository to analyze and fix.
    
    Returns:
    str: A message indicating the result of the operation.
    """
    try:
        # Ensure the local directory exists
        directory = 'local/repo'
        github_token = os.environ.get('github_token')

        file_paths, repo_url = process_user_input_url(repo_path_url)
        primary_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=directory)
        primary_branch = primary_branch.decode('utf-8').strip()
        logging.info(f"Primary branch: {primary_branch}")

        for i, file_path in enumerate(file_paths):
            analysis_list = analyze_file(file_path)
            for analysis in analysis_list:
                if analysis['vulnerability found'] == 'Yes':
                    checkout_branch(primary_branch)

                    base_file_name = os.path.basename(file_path)
                    branch_name = f'fix-{base_file_name}-{i}-{int(time.time())}'
                    create_branch(branch_name)
                    
                    apply_fix(file_path, analysis)
                    commit_message = f'Fix security issues in {base_file_name}'
                    commit_changes(commit_message)
                    
                    push_changes(branch_name, repo_url) 
                    pr_title = f'Security Fix for {base_file_name}-{i}-{int(time.time())}'
                    pr_body = analysis['comment'] + "\n\nVulnerability found and code fix generated using AI powered security tool."
                    open_pull_request(repo_url, branch_name, pr_title, pr_body, directory)
        return "Pull Request Created"
    except Exception as e:
        logging.error(f"Error occurred while analyzing and fixing vulnerabilities: {e}")
        return f"Error occurred while analyzing and fixing vulnerabilities: {e}"

def check_tools_for_agent() -> str:
    """
    This function checks the available tools for the AI agent.
    
    It runs the AI bot with the instruction "What tools do you have available?" and returns the list of available tools.
    
    Returns:
    tools_list (str): The list of available tools for the AI agent.
    """
    try:
        tools_list = run_ai_bot("What tools do you have available?")
        return tools_list
    except Exception as e:
        logging.error(f"Error occurred while checking tools for agent: {e}")
        return f"Error occurred while checking tools for agent: {e}"

def check_tool_viability(task: str) -> str:
    """
    This function checks the viability of a task with the available tools.
    
    It runs the AI bot with the instruction "Can {task} be successfully completed by the following tools: {tools_available}?"
    and returns the result in JSON format.
    
    Parameters:
    task (str): The task to check viability for.
    
    Returns:
    tool_viability (str): The result of the viability check in JSON format.
    """
    try:
        agent_instruction = task
        tools_available = check_tools_for_agent()
        prompt = f"Can {agent_instruction} be successfully completed by the following tools: {tools_available}. If not, what type of new_tool would be needed to perform this task"
        output_format_shot = """
        Please return the response in JSON format
        {"viable tool available": "Yes", "viable tool": "check_slack_secrets"}
        {"viable tool available": "No", "viable tool": "None"}
        """
        tool_viability = llm.predict(prompt + output_format_shot)
        return tool_viability
    except Exception as e:
        logging.error(f"Error occurred while checking tool viability: {e}")
        return f"Error occurred while checking tool viability: {e}"

def load_generated_tools(file_path):
    spec = importlib.util.spec_from_file_location("ai_generated_custom_tools", file_path)
    ai_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ai_tools)
    return [attr for name, attr in ai_tools.__dict__.items() if callable(attr)]

def process_user_task(task: str) -> str:
    try:
        tool_viability = check_tool_viability(task)
        tool_viability_dict = json.loads(tool_viability)
    except Exception as e:
        logging.error(f"Error occurred while checking tool viability: {e}")
        return f"Error occurred while checking tool viability: {e}"

    # Check if a viable tool is available
    if tool_viability_dict.get("viable tool available") == "Yes":
        print(f"Viable tool found: {tool_viability_dict['viable tool']}")
        try:
            result = run_ai_bot(task)
        except Exception as e:
            logging.error(f"Error occurred while running AI bot: {e}")
            return f"Error occurred while running AI bot: {e}"
        return result
    else:
        logging.info(f"Viable tool not found for task: {task}")
        
        # Generate, save, and load the new tool
        new_tool_file = generate_required_tools_code(task)
        generated_tools = load_generated_tools(new_tool_file)
        print("Loaded tools:", generated_tools)

        
        # Re-run the AI bot with the new tools
        try:
            result = run_ai_bot("What tools do you have access to", additional_tools=generated_tools)  # Modify run_ai_bot to accept additional tools
        except Exception as e:
            logging.error(f"Error occurred while running AI bot with new tools: {e}")
            return f"Error occurred while running AI bot with new tools: {e}"
        
        return f"New tool generated and task executed. Result: {result}"


def run_ai_bot(user_input, additional_tools=None):
    """
    This function initializes and runs the AI bot with a set of predefined tools. 
    These tools include checking credentials in a repository, checking for outdated dependencies in a git repository,
    checking for public S3 buckets, checking for AWS users without MFA, and invalidating leaked AWS keys.
    
    Parameters:
    user_input (str): The instruction for the AI bot to execute.
    
    Returns:
    result: The result of the executed instruction.
    """
    if additional_tools is None:
        additional_tools = []
    
    agent_instruction = user_input
    existing_tools = load_tools([], llm=llm)

    tools = existing_tools + additional_tools

    agent= initialize_agent(
        tools + [check_credentials_in_repo] + [check_git_depdency_cves] + [get_public_buckets] + [check_aws_mfa] 
        + [invalidate_aws_key] + [extract_incident_schema] + [check_cve_in_kev] + [threat_model] + [extract_phishing_insights]
        + [analyze_and_fix_vulnerabilities], 
        llm, 
        agent=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        verbose = True)

    result = agent(agent_instruction)
    # result = check_aws_mfa(user_input)
    return result