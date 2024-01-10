import openai
from dotenv import load_dotenv
import os
from typing import Dict
import csv
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time
from queue import Queue

MAX_RETRIES = 5
RATE_LIMIT_WAIT = 10  # seconds


MAX_THREADS = 10  # Adjust as per your requirements

load_dotenv()


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


openai.api_key = OPENAI_API_KEY

topics_with_description = {
    "Unsolicited Ads": "Emails containing unsolicited advertisements.",
    "Vendor Emails": "Service emails from the vendor, including confirmations, notifications, newsletters, updates, and automated messages.",
    "First-Time Refund": "Customers request a refund for the first time without exploring alternatives.",
    "Persistent Refund": "Customers continue to request a refund after exploring alternatives in previous emails.",
    "Shipping Mistake": "Customers complain that the received order differs from what they ordered.",
    "Positive Reviews": "Customers provide positive feedback or reviews.",
    "Complain Size Fit": "Customers provide negative feedback regarding the size fit of the product.",
    "Complain Quality": "Customers provide negative feedback regarding the quality of the product.",
    "Complain Design": "Customers provide negative feedback regarding the design of the product.",
    "Other Negative Reviews": "Customers provide negative feedback for reasons other than size, quality, or design.",
    "Shipping Query": "Customers inquire about the status of their shipment.",
    "Order Cancel": "Customers request to cancel their order.",
    "Order Info Update": "Customers request updates to order information such as size, style, address, or applying a promo code.",
    "Engagement Incentive": "Customers request incentives for engaging with the brand on social media.",
    "Other Customer Support": "Topics that don't fit into any of the existing customer support categories.",
    "Unpaid KOL Collaboration": "Key Opinion Leaders (KOLs) agree to collaborate without additional fees.",
    "Paid KOL Collaboration": "KOLs decline to collaborate without additional fees or ask for them.",
    "KOL Package Received": "KOLs confirm receipt of the package after agreeing to collaborate.",
    "KOL Content Commitment": "KOLs commit to posting content in future on social media featuring the product.",
    "KOL Content Published": "KOLs confirm that they have published the content on social media featuring the product.",
    "KOL Promo Code": "KOLs request a promo code for their followers.",
    "Other KOL Topics": "Topics related to KOLs that don't fit into any of the existing categories."
}

def format_system_prompt(topic_discription, email):
    return f"""
    You are an email label assistant helping an e-commerce team to label a given email.
    An email contains the main message and the quoted thread (the part begin with `>` which can help you understand the context of the email).
    You will be given a list of topics and their descriptions. 
    Please first review and understand the email and the context, compare the main message (not the context) with each descriptions, and select the most relevant topic(s), minimizing the number of topics selected.
    Your output should be in a list of JSON object of fields: topic, confidence_score (0-1), and reason, sorted by confidence_score DESC.
    \n\nHere is the topic-discription list:\n {topic_discription}
    \n\nHere is the email:\n {email}
    """


def needs_processing(email):
    return "\n>" not in email and ("wrote:" in email or "From:" in email)


def reformat_email(email_content):
    if not needs_processing(email):
        return email

    lines = email_content.strip().split("\n")

    # Identify the first message (and its metadata) in the email chain
    metadata_end = 0
    for idx, line in enumerate(lines):
        if line.strip() == "":
            metadata_end = idx
            break

    # Start the reformatted email with the initial metadata and message
    reformatted = "\n".join(lines[:metadata_end+1])

    # Handle the email thread content with different indentation levels
    depth = 0
    for line in lines[metadata_end+1:]:
        stripped = line.strip()

        # Identify change of sender
        if stripped.startswith("On ") and "wrote:" in stripped:
            depth += 1
            reformatted += "\n" + ">" * depth + " " + stripped
        # Skip redundant lines and separators
        elif not stripped or stripped.startswith("--"):
            continue
        else:
            reformatted += "\n" + ">" * depth + " " + stripped

    return reformatted


def topic_discovery_with_gpt_4(topics_with_description: Dict[str, str], email_content: str):
    formated_email = reformat_email(email_content)

    completion = openai.ChatCompletion.create(
        model="gpt-4-0613", 
        messages=[
            {"role": "system", 
             "content": format_system_prompt(topics_with_description, ""),     
             },
            {"role": "user",
             "content": formated_email,
            }
        ]
    )


    return completion.choices[0].message.content

def worker(q, i, row):
    result = process_single_email(i, row)
    q.put(result)


def process_single_email(i, row):
    email_content = row[0]
    print(f"Processing email {i+1}")
    for retry in range(MAX_RETRIES):
        try:
            result = topic_discovery_with_gpt_4(topics_with_description, email_content)
            result_json = json.loads(result)
            topics_list = [topic['topic'] for topic in result_json]
            return [email_content, topics_list, result]
        except Exception as e:
            # You can be more specific in catching exceptions, for instance, if OpenAI provides a specific exception for rate limits.
            if "rate limit" in str(e).lower():
                print(f"Rate limit reached. Waiting for {RATE_LIMIT_WAIT} seconds and then retrying...")
                time.sleep(RATE_LIMIT_WAIT)
            else:
                print(f"Error processing email {i+1}: {e}")
                break  # If it's not a rate-limit issue, we break out of retry loop.
    return [email_content, [], "Error: Unable to process due to repeated failures."]  # Return a default in case of repeated failures.


def process_emails(input_file: str, limit: int, output_file: str):
    results = Queue()
    
    with open(input_file, 'r') as input:
        reader = csv.reader(input)
        header = next(reader)  # Read and store the header row
        
        with ThreadPoolExecutor() as executor:
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                executor.submit(worker, results, i, row)
            
        with open(output_file, 'w', newline='') as output:
            writer = csv.writer(output)
            output_header = ['email content', 'topics', 'result json body']  # Header names for the output columns
            writer.writerow(output_header)  # Write the output header row to the output file
            
            for _ in range(min(limit, len(results.queue))):
                writer.writerow(results.get())
                print(f"Email processed")


email = """
You have 2 projects with the same name (llm-feedback-monitor) under chunyang.shen@afterpay.com and scy0208@gmail.com.
Do these projects contain different data?

On Sat, Aug 19, 2023 at 12:24 PM, C Shen <scy0208@gmail.com> wrote:
Since I have removed chunyang.shen@afterpay.com, now I can only logged as scy0208@gmail.com, I can’t access to the projects under chunyang.shen@afterpay.com any more

On Sat, Aug 19, 2023 at 7:32 AM Alla Tumarkina from Support @ Neon <support@neon.tech> wrote:
Hi there,

I am on the support team over here at Neon. Thank you for reaching out.

There is a project called llm-feedback-monitor under chunyang.shen@afterpay.com

scy0208@gmail.com also has a project with the same name: llm-feedback-monitor

And 2 more projects under scy0208@gmail.com:
vinosensei and taxonomy.

Could you please clarify what disappeared?

Kind regards,
Alla​
--
You received this message because you are subscribed to the Google Groups "support" group.
To unsubscribe from this group and stop receiving emails from it, send an email to support+unsubscribe@neon.tech.
To view this discussion on the web visit https://groups.google.com/a/neon.tech/d/msgid/support/CAKHXwQObxndj4%2BRed9gUY6WyYLwYxQ-DkF26yL%3D_pGL%2BCDmBvQ%40mail.gmail.com.
"""


def main():
    # process_emails("Others.csv", 500, "Others_output.csv")
    print(topic_discovery_with_gpt_4(topics_with_description, email))


if __name__ == '__main__':
    main()