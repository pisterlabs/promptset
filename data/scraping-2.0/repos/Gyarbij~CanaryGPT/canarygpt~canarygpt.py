import openai
import os
import json

try:
    # Get your API key from the environment variables.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    openai.api_key = api_key

    # Get sender from environment variables.
    sender = os.getenv("SENDER_NAME")
    if not sender:
        raise ValueError("Missing SENDER_NAME")

    # Get a list from the environment variables. 
    # Valid input: `$ export RECIPIENT_LIST='["Foo", "bar"]'`
    recipient_list = os.getenv("RECIPIENT_LIST")
    if not recipient_list:
        raise ValueError("Missing RECIPIENT_LIST")
    recipients = json.loads(recipient_list)
    if not all(isinstance(i, str) for i in recipients):
        raise ValueError("RECIPIENT_LIST should be a list of strings")

    def rewrite_email(email_text):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Rewrite the following email in a different way: {email_text}",
            temperature=0.7,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        if response.choices:
            return response.choices[0].text.strip()
        else:
            raise RuntimeError("No response from API")

    email_template = """
    Hi {recipient},

    We are going to move forward with questionable corporate objective. 

    It is designed to achieve maximalist goals while ignoring CSR.

    You are the first to know and as such it is highly confidential.

    Regards,
    {sender}
    """

    for recipient in recipients:
        original_email = email_template.format(recipient=recipient, sender=sender)
        rewritten_email = rewrite_email(original_email)
        print(f"Original email to {recipient}:")
        print(original_email)
        print(f"\nRewritten email to {recipient}:")
        print(rewritten_email)
        print("\n" + "-"*40 + "\n")

except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
