from typing import Dict, List, Union
from googleapiclient.discovery import Resource
from openai import OpenAI

from colorama import Fore
from src.email_evaluation import evaluate_email
import json



processed_emails_details = []  # Global list to store processed email details


def process_email(gmail: Resource, message_info: Dict[str, Union[str, List[str]]], email_data_parsed: Dict[str, Union[str, List[str]]], user_first_name: str, user_last_name: str, client: OpenAI, action: str, processed_emails_file_path: str) -> int:
    # Evaluate email
    if evaluate_email(email_data_parsed, user_first_name, user_last_name, client):
        
        # Prepare email details for tracking
        email_details = {
            'id': message_info['id'],
            'subject': email_data_parsed.get('subject', ''),
            'from': email_data_parsed.get('from', ''),
            'email_contents': email_data_parsed['body'],
            'action': action
        }
        
        if action == 'delete':
            print(Fore.LIGHTYELLOW_EX + "Email is not worth the time, deleting" + Fore.RESET)
            # Delete email
            try:
                gmail.users().messages().delete(userId='me', id=message_info['id']).execute()
                print(Fore.LIGHTGREEN_EX + "Email deleted successfully" + Fore.RESET)
            except Exception as e:
                print(Fore.LIGHTRED_EX + f"Failed to delete email: {e}" + Fore.RESET)
                return 0
        elif action == 'read':
            print(Fore.LIGHTYELLOW_EX + "Email is not worth the time, marking as read" + Fore.RESET)
            # Remove UNREAD label
            try:
                gmail.users().messages().modify(userId='me', id=message_info['id'], body={'removeLabelIds': ['UNREAD']}).execute()
                print(Fore.LIGHTGREEN_EX + "Email marked as read successfully" + Fore.RESET)
            except Exception as e:
                print(Fore.LIGHTRED_EX + f"Failed to mark email as read: {e}" + Fore.RESET)
                return 0
        # Append to the global list
        processed_emails_details.append(email_details)
        # Write to the file after each action
        with open(processed_emails_file_path, 'w') as file:
            json.dump(processed_emails_details, file, indent=4)
        return 1
    else:
        print(Fore.LIGHTBLUE_EX + "Email is worth the time, leaving as unread" + Fore.RESET)
    return 0

def report_statistics(total_unread_emails: int, total_pages_fetched: int, total_marked_as_read: int, model_used: str) -> None:
    print("\n")
    header = "Statistics Report"
    print(f"{Fore.LIGHTCYAN_EX}{header.center(50)}{Fore.RESET}")
    print(f"{Fore.LIGHTCYAN_EX}{'-' * 50}{Fore.RESET}")

    stats = {
        'Total unread emails fetched': total_unread_emails,
        'Total pages fetched': total_pages_fetched,
        'Total emails marked as read': total_marked_as_read,
        'Final number of unread emails': total_unread_emails - total_marked_as_read,
        'Language model used': model_used
    }

    for key, value in stats.items():
        print(f"{Fore.LIGHTYELLOW_EX}{key:<35}{Fore.RESET}{value:<15}")

    print(f"{Fore.LIGHTCYAN_EX}{'-' * 50}{Fore.RESET}")
    footer = "End of Report"
    print(f"{Fore.LIGHTCYAN_EX}{footer.center(50)}{Fore.RESET}")
    print("\n")