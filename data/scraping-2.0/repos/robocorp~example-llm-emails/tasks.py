"""Example bot that receives email messages (for example autoforwarded)
through Robocorp Control Room's email trigger, uses LLM to extract data
and summarize them and then replies to the original email so that bot's
reply is visible in the thread in user's email client.


The use case simulated here is to act as a helper to a B2B payment
collections, but by editing the prompts and the code, you can use this
for any other use case, too."""

from robocorp.tasks import task
from robocorp import vault, workitems
import openai
import json
import traceback
from email import message_from_file
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Header


PROMPT_TEMPLATE = """Acting as a helper to a payment collections agent for a B2B company, your task is to get the relevant data out of the email discussion with the customer. The email thread is about unpaid invoices.

Your specific task is to return data per each separate invoice in the thread, indicating what customer has responded to each of the invoices payment status. Produce a JSON-formatted response only.

This is the email conversation between the agent and the customer:
###--DISCUSSION--###

The response must be in the JSON format containing the following keys and values:
{
"summary": "summary of the entire conversation in max 3 sentences",
"account_id": "account id of the customer, typically found in the subject line",
"invoices": "list of JSON elements that have following data for each invoice covered in the discussion: invoice_id, total_value, currency, status (based on the information on the discussion, the status can be one of the following 'paid', 'payment_promised', 'dispute', 'request_info', 'waiting_approval' or 'other', see detailed descriptions of these statuses later in the prompt), promised_payment_date (the date customer has indicated the payment will be made in the format YYYY-MM-DD if status is payment_promised, otherwise the value is empty string) and summary (this should contain an invoice specific summary of what has customer said specifically about this invoice in one sentence",
"suggested_reply": "recommend a reply to the customer to his last message based on the information in the discussion so far, with the goal of providing the customer with the information he needs to proceed with the payment(s). Use placeholders for content that you don't have available.",
}

Description of the statuses in the above JSON format:
- paid: customer has indicated that this invoice has already been paid
- payment_promised: customer indicates an intention that the invoice will be paid at a certain date. In this case enter the date in promised_payment_date key in the JSON.
- dispute: customer disputes or rejects the invoice for any reason.
- request_info: customer has asked for more information such as copy of the invoice
- waiting_approval: customer indicates that their business owner or buyer has still to approve the invoice before the payment can be scheduled
- other: anything other than above

Make sure that the `invoices` list will contain the correct promised payment date, if it is mentioned that the invoice will be or was paid at a specific date, or empty string otherwise.

Please give only the properly structured JSON in the response (not code, not comments, not anything else):
"""

SYSTEM_PROMPT = """You are an assistant that deals with payment collections. Your role is to extract structured data from the email conversations and suggest the next best replies.
"""

def initialize():
    """Initialize the desired LLM and Email API clients.
    This example uses OpenAI API and SendGrid, which are configured
    in the Robocorp Vault."""

    openai_secrets_container = vault.get_secret("OpenAI")
    model = "gpt-4"
    openai.api_key = openai_secrets_container["key"]

    # Set up SendGrid API client
    sg_secrets_container = vault.get_secret("Sendgrid")
    from_email = sg_secrets_container["FROM_EMAIL"]
    sg_client = SendGridAPIClient(sg_secrets_container["SENDGRID_API_KEY"])

    return sg_client, from_email, model


def create_prompt(discussion: str) -> str:
    """Construct the prompt for the LLM using template."""

    # It's also possible to use Robocorp Asset Storage for templates,
    # which allows editing them without releasing a new version
    # of the robot.
    # PROMPT_TEMPLATE = storage.get_asset("llm-prompt-template")

    # Replace the discussion placeholder with the actual discussion
    prompt = PROMPT_TEMPLATE.replace("--DISCUSSION--", discussion)
    return prompt


def create_system_prompt() -> str:
    """Return the system prompt for the LLM"""

    # It's also possible to use Robocorp Asset Storage for templates,
    # which allows editing them without releasing a new version
    # of the robot.
    # SYSTEM_PROMPT = storage.get_asset("llm-system-prompt")
    return SYSTEM_PROMPT


def construct_email_reply(response: str) -> str:
    """Construct the email reply based on the response from the LLM."""
    try:
        json_data = json.loads(response["choices"][0]["message"]["content"])
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", str(e))
        return ""
    except Exception as e:
        print("An error occurred:", str(e))
        return ""

    reply = f"""
{get_css_template()}
<body>
<h2>SUMMARY</h2>
{json_data.get("summary", "No summary was returned")}

<h2>SUGGESTED REPLY</h2>
{json_data.get("suggested_reply", "No suggested reply was returned")}

<h2>INVOICES</h2>
<table>
<thead>
    <tr>
        <th class="resizable">Invoice ID</th>
        <th class="resizable">Value</th>
        <th class="resizable">Status</th>
        <th class="resizable">Payment promised</th>
        <th class="resizable">Summary</th>
        <th class="resizable">Action</th>
    </tr>
</thead>
<tbody>
"""
    for invoice in json_data.get("invoices", []):
        reply += f"""
        <tr>
            <td>{invoice.get("invoice_id", "NO ID")}</td>
            <td>{invoice.get("total_value", "NO VALUE")} {invoice.get("currency", "")}</td>
            <td>{invoice.get("status", "NO STATUS")}</td>
            <td>{invoice.get("promised_payment_date", "NO PROMISED DATE")}</td>
            <td>{invoice.get("summary", "NO SUMMARY")}</td>
            <td><a href="https://www.w3.org/Provider/Style/dummy.html">Update AR</a></td>
        </tr>
"""

    reply += "</tbody></table><br /><p>Bot Generated Reply Ends Here</p>"

    return reply


def get_css_template():
    """Return the CSS template for the email reply."""
    return """
<!DOCTYPE html>
<html>
<head>
  <style>
    /* CSS styles */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    table {
      width: 100%;
      font-family: 'Roboto', sans-serif;
      border-collapse: collapse;
    }

    thead th {
      padding: 12px;
      text-align: left;
      background-color: #f2f2f2;
      color: #333333;
      font-weight: bold;
      border-bottom: 2px solid #dddddd;
    }

    tbody td {
      padding: 12px;
      border-bottom: 1px solid #dddddd;
    }

    tbody tr:nth-child(even) {
      background-color: #f9f9f9;
    }

    tbody tr:hover {
      background-color: #ebebeb;
    }

    /* Optional: Resizable columns (requires JavaScript) */
    th.resizable {
      position: relative;
      cursor: col-resize;
    }

    th.resizable:after {
      content: "";
      position: absolute;
      top: 0;
      right: -4px;
      bottom: 0;
      width: 8px;
      background-color: #dddddd;
      z-index: 1;
    }

    th.resizable:hover:after {
      background-color: #cccccc;
    }
  </style>
</head>
"""


@task
def process_emails():
    """Read email, do LLM magic and reply to the sender."""

    # Inititalize it all
    sg, from_email, model = initialize()
    sys_prompt = create_system_prompt()

    # Loop through input work items, there should be only emails in them.
    # There most likely should be only one item and one email,
    # but we'll handle multiple just in case.
    for item in workitems.inputs:
        # Get email from work item, and it's header details that are needed later
        try:
            email = item.email()
            inReplyTo = item.payload["email"]["inReplyTo"]
            references = " ".join(item.payload["email"]["references"])
        except Exception as e:
            print(f"Error reading email from payload: {e}")
            break

        prompt = create_prompt(email.text)

        response = openai.ChatCompletion.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        # Print some debug info
        print(f"*********** TOKEN USAGE *************\n{response['usage']}\n\n")
        print(
            f"*********** RESPONSE CONTENT *************\n{response['choices'][0]['message']['content']}\n\n"
        )

        # Create a reply email. Note that the original email content
        # is not added in this email at all. You could do that if you want.
        reply = construct_email_reply(response)
        message = Mail(
            from_email=from_email,
            to_emails=email.from_.address,
            subject="Re: " + email.subject,
            html_content=reply,
        )

        # Add headers to put the email in the same thread as the original.
        message.header = [
            Header("in_reply_to", inReplyTo),
            Header("References", references),
        ]

        # SEND IT!
        try:
            response = sg.send(message)
        except Exception as e:
            print(traceback.format_exc())