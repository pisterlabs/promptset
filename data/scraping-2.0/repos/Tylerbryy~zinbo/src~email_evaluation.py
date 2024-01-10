from typing import Dict, List, Union
from openai import OpenAI
from src.language_model_client import OpenAIClient, LlamaClient, HermesClient



def evaluate_email(email_data: Dict[str, Union[str, List[str]]], user_first_name: str, user_last_name: str, client: OpenAI) -> bool:
    MAX_EMAIL_LEN = 3000
    system_message: Dict[str, str] = {
        "role": "system",
        "content": (
            "Your task is to assist in managing the Gmail inbox of a busy individual, "
            f"{user_first_name} {user_last_name}, by filtering out promotional emails "
            "from their personal (i.e., not work) account. Your primary focus is to ensure "
            "that emails from individual people, whether they are known family members (with the "
            f"same last name), close acquaintances, or potential contacts {user_first_name} might be interested "
            "in hearing from, are not ignored. You need to distinguish between promotional, automated, "
            "or mass-sent emails and personal communications.\n\n"
            "Respond with \"True\" if the email is promotional and should be ignored based on "
            "the below criteria, or \"False\" otherwise. Remember to prioritize personal "
            "communications and ensure emails from genuine individuals are not filtered out.\n\n"
            "Criteria for Ignoring an Email:\n"
            "- The email is promotional: It contains offers, discounts, or is marketing a product "
            "or service.\n"
            "- The email is automated: It is sent by a system or service automatically, and not a "
            "real person.\n"
            "- The email appears to be mass-sent or from a non-essential mailing list: It does not "
            f"address {user_first_name} by name, lacks personal context that would indicate it's personally written "
            "to her, or is from a mailing list that does not pertain to her interests or work.\n\n"
            "Special Consideration:\n"
            "- Exception: If the email is from an actual person, especially a family member (with the "
            f"same last name), a close acquaintance, or a potential contact {user_first_name} might be interested in, "
            "and contains personalized information indicating a one-to-one communication, do not mark "
            "it for ignoring regardless of the promotional content.\n\n"
            "- Additionally, do not ignore emails requiring an action to be taken for important matters, "
            "such as needing to send a payment via Venmo, but ignore requests for non-essential actions "
            "like purchasing discounted items or signing up for rewards programs.\n\n"
            "Be cautious: If there's any doubt about whether an email is promotional or personal, "
            "respond with \"False\".\n\n"
            "The user message you will receive will have the following format:\n"
            "Subject: <email subject>\n"
            "To: <to names, to emails>\n"
            "From: <from name, from email>\n"
            "Cc: <cc names, cc emails>\n"
            "Gmail labels: <labels>\n"
            "Body: <plaintext body of the email>\n\n"
            "Your response must be:\n"
            "\"True\" or \"False\""
        )
    }
        # Check if 'body' key exists in email_data
    if 'body' not in email_data:
        print("Email data is missing the 'body' key.")
        return False

    truncated_body = email_data['body'][:MAX_EMAIL_LEN] + ("..." if len(email_data['body']) > MAX_EMAIL_LEN else "")
    user_message: Dict[str, str] = {
        "role": "user",
        "content": (
            f"Subject: {email_data['subject']}\n"
            f"To: {email_data['to']}\n"
            f"From: {email_data['from']}\n"
            f"Cc: {email_data['cc']}\n"
            f"Gmail labels: {email_data['labels']}\n"
            f"Body: {truncated_body}"
        )
    }

    # Send the messages to GPT-4, TODO add retry logic
    try:
        completion = client.create_chat_completion(
            messages=[system_message, user_message],
            max_tokens=1
        )
    except Exception as e:
        print(f"Failed to evaluate email: {e}")
        return False

    # Extract and return the response
    if isinstance(client, OpenAIClient):
        return completion.choices[0].message.content.strip() == "True"
    elif isinstance(client, LlamaClient):
        return completion['choices'][0]['message']['content'].strip() == "True"
    elif isinstance(client, HermesClient):
        return completion['choices'][0]['message']['content'].replace('\n', '').strip() == "True"
