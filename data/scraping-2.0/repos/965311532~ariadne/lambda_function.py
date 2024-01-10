import json

import markdown
import urllib3
from openai import OpenAI

from ariadne import Ariadne, AriadnePrompt
from config import ARIADNE_EMAIL_ADDRESS, ZAPIER_WEBHOOK_URL


def lambda_handler(event, context):
    """Relay email webhook from Zapier to RelevanAI."""

    # Print the event in the logs for debugging
    print(f"DEBUG: Event: {event}")

    # Istantiate the OpenAI client
    openai_client = OpenAI()

    # Istantiate Ariadne
    ariadne = Ariadne(openai_client=openai_client, debug=event.get("debug", False))

    # Send the email to Ariadne
    email = {**event, "from_": event.get("from")}  # Rename the "from" key to "from_"
    answer = ariadne.get_reply(message=AriadnePrompt(email=email).build())

    # Print answer in the logs for debugging
    print(f"DEBUG: Answer: {answer}")

    # If the answer is "NO_RESPONSE", return
    if answer != "NO_RESPONSE":
        # Remove empty and Ariadne's email address from the cc
        cc = (event.get("cc", "") + "," + event.get("to")).split(",")
        cc = ",".join([c for c in cc if c != "" and c != ARIADNE_EMAIL_ADDRESS])

        # Call the zapier webhook to send the email
        http = urllib3.PoolManager()
        http.request(
            method="POST",
            url=ZAPIER_WEBHOOK_URL,
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "thread_id": event.get("thread_id"),
                    "from": ARIADNE_EMAIL_ADDRESS,
                    "to": event.get("from"),
                    "cc": cc,
                    "subject": f"Re: {event.get('subject')}",
                    "body": markdown.markdown(answer),  # Convert the answer to HTML
                }
            ),
        )

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "success"}),
    }
