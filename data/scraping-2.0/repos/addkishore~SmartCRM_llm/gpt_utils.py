import os

import openai
from apify_client import ApifyClient

from dotenv import load_dotenv

load_dotenv()
apify_client = ApifyClient(os.getenv("APIFY_API_KEY"))


def summarize_email(user: str, message: str) -> str:
    """
    Returns a 1-2 sentence summary of the given message.

    :param user: The user, whom which the system will refer to as "you".
    :param message: The email contents.
    """

    summary = (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"""I am {user}. Refer to all instances of {user} as "you". Summarize the given emails in 2 short sentences or fewer.
                            
                            Example summary 1: You sent a message to person B asking them for an internship. You were inspired by their talk at the YC event and gave them your contact information.
                            Example summary 2: John Doe responded to you saying that they were impressed with your resume and would like to set up a meeting with you.
                            """,
                },
                {
                    "role": "user",
                    "content": f"{message}",
                },
            ],
            temperature=0.05,
        )
        .choices[0]  # type: ignore
        .message.content
    )

    return summary


def summarize_webpage(url: str):
    # Prepare the actor input
    run_input = {
        "instructions": "give me a summary of the article on this website",
        "proxyConfiguration": {"useApifyProxy": True},
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Page title"},
                "description": {"type": "string", "description": "Page description"},
            },
            "required": ["title", "description"],
        },
        "startUrls": [{"url": url}],
        "useStructureOutput": False,
        "globs": [],
        "maxCrawlingDepth": 0,
        "maxPagesPerCrawl": 10,
    }

    # Run the actor and wait for it to finish
    run = apify_client.actor("drobnikj/gpt-scraper").call(run_input=run_input)

    # Fetch and print actor results from the run's dataset (if there are any)
    for item in apify_client.dataset(run["defaultDatasetId"]).iterate_items():  # type: ignore
        return item["answer"]


def summarize_company_innovations(text: str):
    summary = (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"""Give me a list of the new innovations based \
on the text and explanations of what the innovation is. It should follow the \
following format:
Product 1: explanation
Product 2: explanation
etc.""",
                },
                {
                    "role": "user",
                    "content": f"{text}",
                },
            ],
            temperature=0.05,
        )
        .choices[0]  # type: ignore
        .message.content
    )

    return summary


def generate_response_email_from_messages(user: str, messages: str):
    summary = (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"""I am {user}. Refer to all instances of \
{user} as "you". I will give you a list of emails. Based on the email \
contents, give me a potential response email action.
Example 1: If it seems that the previous messages have come to a close, the \
response email should be an update email such as this one:
Hi,

I hope you are well. [Make a reference to the previous emails. Should be thanking them].

Since we last connected, I've [Talk about new projects I've worked on].

Look forward to staying in touch, and thank you again for helping me during this journey! 

Best,
{user}

Example 2: If it seems that the previous messages are trying to set up a meeting \
time, the response email should be a meeting setup email such as this one:
Hi,

I just wanted to follow up on this. Let me know what days and times work best \
for you and I can send a calendar invite.

Best,
{user}
""",
                },
                {
                    "role": "user",
                    "content": messages,
                },
            ],
            temperature=0.05,
        )
        .choices[0]  # type: ignore
        .message.content
    )

    return summary
