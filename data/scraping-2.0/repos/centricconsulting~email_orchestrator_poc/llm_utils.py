import openai
import os
from dotenv import load_dotenv
import json
from file_utils import append_text_to_file

load_dotenv()


def analyze_email(message, target_receiver):
    # Define the intro and prompt templates
    intro = f"Analyze and summarize following email sent to {target_receiver} to determine its intent for the recipient only:"

    prompt_template = f"""\
        If the email is only providing information, or informative, provide the response in the following JSON format:
        {{
          "action_type": "inform",
          "item_name": "",
          "Notes": "[SUMMARY_OF_INFORMATION]",
          "Due Date": ""
        }}

        If the email requires or requests an action by the receiver respond in JSON format:
        {{
          "action_type": "action",
          "item_name": "[SUMMARY_OF_ACTION_REQUIRED]",
          "Notes": "[SUMMARY_OF_CONTEXT_AND_BULLET_LIST_OF_ACTIONS_REQUESTED]",
          "Due Date": "[DUE_DATE_IF_PRESENT]"
        }}

        If the email is an out of office, or OOTO, notice respond in JSON format:
        {{
          "action_type": "inform",
          "item_name": "",
          "Notes": "[PERSON_WHO_IS_OUT_AND_DATE_OF_RETURN]",
          "Due Date": ""
        }}
        
        If the email does not have enough information to determine intent respond in JSON format:
        {{
          "action_type": "action",
          "item_name": "Review Email - Intent Unknown",
          "Notes": "Review Email - Intent Unknown",
          "Due Date": ""
        }}

        Do not include the {target_receiver}'s response in the summary.  

        The Due Date must be in the format of YYYY-mm-dd if present
        Only reply in the provided json format.
    """
    prompt = f"{intro}\n{prompt_template}"

    # only look at current email and 1 prior
    trimmed_email = (" ".join(message.body.split("From: ")[:2]))
    if os.environ['API_TYPE'] == 'openai':
        result = openai.ChatCompletion.create(
            model=os.environ["CHAT_MODEL"],
            messages=[{"role": "assistant", "content": prompt},
                      {"role": "user", "content": f"Here is the email: {trimmed_email}"}],
            temperature=0.0,
            api_key=os.environ["OPENAI_API_KEY"]
        )
    elif os.environ['API_TYPE'] == 'azure':
        result = openai.ChatCompletion.create(
            messages=[{"role": "assistant", "content": prompt},
                      {"role": "user", "content": f"Here is the email: {trimmed_email}"}],
            temperature=0.0,
            # this links to a single model, so in Azure OpenAI we don't specify the model, just the deployment
            deployment_id=os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_type='azure',
            api_base=os.environ['AZURE_OPENAI_API_BASE'],
            api_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
    else:
        print("Error in config setting.  Cannot tell if you want Azure or OpenAI endpoint")
        result = None
    return result


def extract_data_from_response(response):
    res = ""
    try:
        res = json.loads(response["choices"][0]["message"]["content"].strip())
    except json.JSONDecodeError:
        append_text_to_file("errors.txt", f"ERROR PROCESSING LLM RESPONSE:\n{response}")
        print(f"ERROR PROCESSING LLM RESPONSE:\n{response}")
    return res


def log_token_usage(response):
    append_text_to_file("token_usage.txt", f"{response['usage']['prompt_tokens']},"
                                           f"{response['usage']['completion_tokens']},"
                                           f"{response['usage']['total_tokens']}")
