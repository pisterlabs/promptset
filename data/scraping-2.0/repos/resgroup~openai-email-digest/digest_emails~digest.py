import os
import json
import openai
import tiktoken
import json
from datetime import datetime


def read_last_run_date():
    file_path = os.path.join(Config.EMAIL_DIRECTORY, "last_run.json")
    try:
        # Open the JSON file for reading
        with open(file_path, "r") as file:
            data = json.load(file)
            # Parse the lastRunDate from the JSON data
            last_run_date_str = data["lastRunDate"]
            # Convert the date string into a datetime object
            last_run_date = datetime.strptime(last_run_date_str, "%Y-%m-%d %H:%M:%S")
            return last_run_date
    except FileNotFoundError:
        print("Last Run File Not Found")
        return None
    except json.JSONDecodeError:
        print("Last Run File Does Not Contain Valid JSON")
        return None
    except KeyError:
        print("Last Run File Does Not Contain lastRunDate")
        return None
    except ValueError as e:
        print(f"Last Run File Contains Invalid Date: {e}")
        return None


class Config:
    @classmethod
    def read(cls):
        cls.OPENAI_MODEL_SMALL = os.getenv("OPENAI_MODEL_SMALL")
        cls.OPENAI_MODEL_LARGE = os.getenv("OPENAI_MODEL_LARGE")
        cls.EMAIL_DIRECTORY = os.getenv("EMAIL_DIRECTORY")

        if not cls.OPENAI_MODEL_SMALL:
            raise Exception("OPENAI_MODEL_SMALL is not configured")

        if not cls.OPENAI_MODEL_LARGE:
            raise Exception("OPENAI_MODEL_LARGE is not configured")

        if not cls.EMAIL_DIRECTORY:
            raise Exception("EMAIL_DIRECTORY is not configured")

        cls._configure_openai()

    @classmethod
    def _configure_openai(cls):
        # OpenAI API Key
        openai.api_type = "azure"
        openai.api_version = "2023-07-01-preview"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_ENDPOINT")


def _truncate_text_to_token_limit(text, model):
    limits = {
        "gpt35-small": 128000,
        "gpt35-large": 128000,
    }

    limit = limits[model]

    encoding = tiktoken.encoding_for_model("gpt-35-turbo")

    tokens = encoding.encode(text)

    if len(tokens) > limit:
        print(f"Truncating to {model} token limit of {limit}...")
        truncated_tokens = tokens[:limit]
        return encoding.decode(truncated_tokens)
    else:
        return text


class Email:
    def __init__(self, file_path):
        self.file_path = file_path

        with open(self.file_path, "r", encoding="utf-8-sig") as file:
            data = json.load(file)

        self.subject = data["Subject"]
        self.body = data["Body"]
        self.sender_name = data["SenderName"]
        self.sender_email_address = data["SenderEmailAddress"]
        self.received_time = data["ReceivedTime"]

        if "Summary" not in data or "Category" not in data:
            print("Processing: " + self.subject)

            args = self._summarise_email()

            data["Summary"] = args["summary"]
            data["Category"] = args["category"]

            if "suggested_action" in args:
                data["SuggestedAction"] = args["suggested_action"]

            # Open the JSON file for writing
            with open(self.file_path, "w") as file:
                # Write the updated data to the file
                json.dump(data, file, indent=4)

        self.summary = data["Summary"]
        self.category = data["Category"]
        self.suggested_action = data.get("SuggestedAction")

    def _summarise_email(self):
        model = Config.OPENAI_MODEL_SMALL

        if not model:
            raise Exception("OPENAI_MODEL_SMALL is not configured")

        truncated_body = _truncate_text_to_token_limit(self.body, model)

        content = (
            f"Subject: {self.subject}\n\n"
            f"From: {self.sender_name} ({self.sender_email_address})\n\n"
            f"Received: {self.received_time}\n\n"
            f"Message: \n{truncated_body}"
        )

        messages = []

        messages.append(
            {
                "role": "system",
                "content": "Here is an email body. Can you summarise and categorise it? Use the summarise_email function.",
            }
        )

        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

        functions = [
            {
                "name": "save_summary",
                "description": (
                    "save a summary of an email with a categorisation as either action, info or ignore. "
                    "If categorised as an action generate a suggested action. "
                    "Please ensure:\n"
                    "- Reminder emails from business systems are categorised as actions. "
                    "- Reminder emails about tasks are categorised as actions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A succinct summary of the email",
                        },
                        "category": {
                            "type": "string",
                            "description": "The email category",
                            "enum": ["action", "info", "ignore"],
                        },
                        "suggested_action": {
                            "type": "string",
                            "description": "A suggested action (if any) to take in response to the email.",
                        },
                    },
                    "required": ["summary", "category"],
                },
            }
        ]

        response = openai.ChatCompletion.create(
            deployment_id=model,
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        output = response["choices"][0]["message"]["function_call"]

        name = output["name"]

        if name.lower() != "save_summary":
            raise Exception(f"Unexpected function called: {name}")

        args = json.loads(output["arguments"])

        return args


def _summarise_emails_by_category(emails):
    model = Config.OPENAI_MODEL_LARGE

    content = "|Item|Subject|Summary|Category|Suggested Action|\n"
    content += "|---|---|---|---|---|\n"

    for index, email in enumerate(emails):
        content += f"|{index}|{email.subject}|{email.category}|{email.summary}|{email.suggested_action}|\n"

    messages = []

    messages.append(
        {
            "role": "system",
            "content": (
                "Here is an table of summarised and categorised emails.\n"
                "Can you please provide an overarching summary as follows:\n"
                "1. Actions that are needed\n"
                "2. Key information the user should know (but no action required)\n"
                "3. Emails that can be ignored/deleted"
            ),
        }
    )

    messages.append(
        {
            "role": "user",
            "content": content,
        }
    )

    response = openai.ChatCompletion.create(
        deployment_id=model,
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response["choices"][0]["message"]["content"]


def _prepare_emails():
    emails = []

    for folder in os.listdir(Config.EMAIL_DIRECTORY):
        day_folder = os.path.join(Config.EMAIL_DIRECTORY, folder)
        if os.path.isdir(day_folder):
            for file_name in os.listdir(day_folder):
                if file_name.endswith(".json"):
                    file_path = os.path.join(day_folder, file_name)
                    email = Email(file_path=file_path)
                    emails.append(email)

    return emails


def get_summary():
    Config.read()

    emails = _prepare_emails()

    summary = _summarise_emails_by_category(
        emails=emails,
    )

    last_run = read_last_run_date()

    if last_run:
        summary += f"\n\n[Emails last exported from outlook on: {last_run}]"
    else:
        summary += (
            "\n\n[Could not determine when emails were last exported from outlook]"
        )

    return summary
