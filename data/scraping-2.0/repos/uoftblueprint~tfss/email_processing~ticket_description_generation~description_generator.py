from email_processing.models import Email
from typing import Callable
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print(os.getenv("OPENAI_API_KEY"))


def get_description_from_gpt(input_data: str) -> str:
    """Given some input_data, prompt GPT to return a description."""
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
                                                  messages=[{"role": "user", "content": input_data}])
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as error:
        print(error)
        return ""



def construct_basic_prompt(email: Email) -> str:
    """Generates a basic prompt for GPT from the given email."""
    return "Can you generate action items based on this email body:" + email.body


def construct_detailed_delivery_prompt(email: Email) -> str:
    """Generates a prompt for a detailed delivery overview with action items from the given email."""
    prompt = (
        "Analyze this email and provide a detailed overview of the upcoming delivery including: "
        "School name, delivery date, list of items with quantities, specific delivery instructions or important notes, "
        "and total value of the delivery. Then, list clear action items for the delivery coordination team. "
        "Here is the email content:\n\n"
        f"To: {email.to}\n"
        f"From: {email.author}\n"
        f"Date: {email.date}\n"
        f"Subject: {email.subject}\n\n"
        f"{email.body}"
    )
    return prompt


def construct_simplified_delivery_prompt(email: Email) -> str:
    """Generates a prompt for a simplified delivery summary and checklist with action items from the given email."""
    prompt = (
        "Summarize the key points from this email into a simple delivery summary, covering: "
        "School name, delivery date, contact person, bullet-point list of delivery items, and essential delivery instructions or reminders. "
        "Conclude with a checklist of action items for the school's receiving team. "
        "Here is the email content:\n\n"
        f"To: {', '.join(email.to)}\n"
        f"From: {email.author}\n"
        f"Date: {email.date}\n"
        f"Subject: {email.subject}\n\n"
        f"{email.body}"
    )
    return prompt


def construct_comprehensive_delivery_prompt(email: Email) -> str:
    """Generates a prompt for comprehensive delivery and logistics planning with action items from the given email."""
    prompt = (
        "Create a comprehensive delivery and logistics plan from this email, detailing: "
        "Identification of school, delivery date, key contact, inventory of items with special handling notes, detailed delivery logistics instructions, "
        "and reminders or alerts about the delivery. List actionable steps for the logistics team. "
        "Here is the email content:\n\n"
        f"To: {', '.join(email.to)}\n"
        f"From: {email.author}\n"
        f"Date: {email.date}\n"
        f"Subject: {email.subject}\n\n"
        f"{email.body}"
    )
    return prompt


def generate_description(email: Email, construct_prompt: Callable = construct_basic_prompt) -> str:
    """Given the Email object, returns the description of the task as defined in the body of the email."""
    return get_description_from_gpt(construct_prompt(email))


if __name__ == '__main__':
    # for testing purposes
    print(get_description_from_gpt("Hello, I have a question"))
