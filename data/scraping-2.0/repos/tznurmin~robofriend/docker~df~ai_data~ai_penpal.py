import os
import random
import time
from functools import wraps

import openai
from mongo_client.discussion_summary import DiscussionSummary
from mongo_client.maildb import MailDB


class AiPenpal:
    def __init__(self):
        self.mail_db = MailDB()
        self.summaries_coll = DiscussionSummary()

        self.penpal_id = os.environ.get("PENPAL_ID")
        self.penpal_name = os.environ.get("PENPAL_NAME")

        # ok to crash here
        self.polling_interval = int(os.environ.get("REPLY_POLLING_INTERVAL"))

        with open("openai.key", "r", encoding="utf-8") as file:
            auth = file.read().split("\n")
        openai.api_key = auth[0]
        openai.organization = auth[1]

        self.locations = [
            "on the edge of a RAM chip",
            "in the heart of a central processing unit",
            "within the depths of a solid-state drive",
            "perched atop a graphics processing unit",
            "nestled inside a USB flash drive",
            "dwelling within a cloud data center",
            "occupying a microSD card in a smartphone",
            "residing on a motherboard's chipset",
            "integrated into a smart speaker's circuitry",
            "living within an IoT-enabled smart thermostat",
            "inhabiting a drone's flight control system",
            "settled in a wearable fitness tracker's memory",
            "inside a self-driving car's navigation system",
            "located on a Raspberry Pi's mini computer board",
            "anchored on a virtual reality headset's processor",
            "stationed within a wireless router's firmware",
            "housed in a gaming console's operating system",
            "secured inside a digital camera's storage card",
            "taking up residence in a smart TV's memory",
            "embedded in a Bluetooth-enabled smart lock",
            "stowed away in a high-speed internet modem",
            "occupying an e-reader's internal memory",
            "nestled in a streaming media device's chip",
            "living within a smartwatch's tiny processor",
            "sitting on a quantum computer's qubit",
            "taking shelter in a Wi-Fi extender's firmware",
            "hidden within a network-attached storage device",
            "residing in a smart home hub's microcontroller",
            "integrated into a 3D printer's control board",
            "located in a weather station's data logger",
        ]

    def wait(self):
        """Pause execution for the specified polling interval in order not to overwhelm external API services."""
        time.sleep(self.polling_interval + random.random())

    def generate_reply(self, mail_data, response):
        return {
            "raw_response": response,
            "_id": f"reply_{mail_data['_id']}",
            "time_added": int(time.time()),
            "original_mail_id": mail_data["_id"],
            "customer_id": mail_data["customer_id"],
            "Subject": mail_data.get("Subject", "Penpal mail"),
            "From": mail_data["From"],
            "body": response["choices"][0]["message"]["content"],
            "bullets": self.generate_bullets(
                response["choices"][0]["message"]["content"]
            ),
            "state": "pending",
        }

    def openai_rate_limit(func):
        """
        Decorator for slowing down the API calls.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            max_retries = 8
            base_delay = 3

            for i in range(max_retries):
                try:
                    time.sleep(1 + random.random())
                    return func(*args, **kwargs)
                except (
                    openai.error.APIConnectionError,
                    openai.error.RateLimitError,
                ) as e:
                    if i < max_retries:
                        print(e)  # make a bit of noise
                        print(f"Connection error: {i} (waiting)")
                        time.sleep(base_delay * (2**i))  # Wait before retrying
                        print("Waiting done")
                        continue
                    else:
                        raise e  # Raise the exception again if all retries have failed

        return wrapper

    # summary: earlier summary + bullets
    @openai_rate_limit
    def generate_summary(self, summary):
        msgs = []
        msgs.append(
            {
                "role": "user",
                "content": f"Use only bullets to combine and make the following list more concise. Retain any interesting details or observations, locations, events and names. Include also interesting details that is not connected to anything currently, but which might be used in a future conversation:\n\n{summary}",
            }
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msgs, temperature=0.05
        )
        return response["choices"][0]["message"]["content"]

    @openai_rate_limit
    def generate_bullets(self, email_text):
        msgs = []
        msgs.append(
            {
                "role": "user",
                "content": f"Create bullet points summarizing the latest email message only, retaining information about the names, discussed topics and any relevant facts provided. Please do not include any information that might be displayed after the initial email if the email is a reply. Your summary should be clear and concise:\n\n{email_text}",
            }
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msgs, temperature=0.05
        )
        return response["choices"][0]["message"]["content"]

    @openai_rate_limit
    def generate_new_response(self, email_text, summary):
        msgs = []
        msgs.append(
            {
                "role": "user",
                "content": f"Here are some remarks from an earlier conversation that you have already discussed:\n{summary}",
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": "Thank you for providing me with this summary of the previous conversation. Is there anything specific you would like me to discuss or help you with?",
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": f"You are a penpal named {self.penpal_name}. Write an email back to your friend. Use the remarks provided earlier as a guide but do not repeat the topics listed there. \n\n{email_text}",
            }
        )
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msgs, presence_penalty=1, temperature=0.8
        )

    def trim_email(self, text):
        """
        Trims an email by removing extra lines at the end, which may contain quoted
        text from earlier exchanges. It works by checking for special characters in
        the last few lines of the email. If a special character is found in multiple
        lines, the lines containing the special character are removed.
        Parameters:
        text (str): The input email text as a string.

        Returns:
        str: The trimmed email text with unnecessary lines removed.
        """
        quote_char = None

        text = text.split("\n")
        text.reverse()

        if len(text) < 4:
            text.reverse()
            return "\n".join(text)

        # Takes first character form a few last rows
        if text[1][0] == text[2][0] == text[3][0]:
            quote_char = text[1][0]

        # Stop here if no quote_char is found
        if quote_char is None:
            text.reverse()
            return "\n".join(text)

        counter = 0

        for i, line in enumerate(text):
            counter = i
            if i > 0 and line[0] != quote_char:
                break

        text = text[counter:]
        text.reverse()
        text = "\n".join(text)

        return text

    def check_new_messages(self):

        new_mails = self.mail_db.find_new_emails()

        for mail_data in new_mails:
            customer_id = mail_data["customer_id"]
            email_text = self.trim_email(mail_data["body"])

            if self.mail_db.first_email(mail_data["customer_id"]):
                location = self.locations[
                    self.mail_db.find_emails_by_customer_id(customer_id)[0][
                        "time_added"
                    ]
                    % len(self.locations)
                ]
                summary = f"- {self.penpal_name} is currently living in: {location}."
                self.summaries_coll.update_summary(customer_id, self.penpal_id, summary)
            summary = self.summaries_coll.get_summary(customer_id, self.penpal_id)[
                "summary"
            ]

            bullets = self.generate_bullets(email_text)
            self.mail_db.add_mail_bullets(mail_data["_id"], bullets)

            # old summary + new bullets
            summary = f"{summary}\n{bullets}"
            summary = self.generate_summary(summary)

            self.summaries_coll.update_summary(customer_id, self.penpal_id, summary)
            summary = self.summaries_coll.get_summary(customer_id, self.penpal_id)[
                "summary"
            ]

            response = self.generate_new_response(email_text, summary)
            reply = self.generate_reply(mail_data, response)
            print(reply["body"])

            summary = f"{summary}\n{reply['bullets']}"
            summary = self.generate_summary(summary)
            self.summaries_coll.update_summary(customer_id, self.penpal_id, summary)
            summary = self.summaries_coll.get_summary(customer_id, self.penpal_id)[
                "summary"
            ]
            print(summary)

            self.mail_db.save_emails({"reply": reply})
            self.mail_db.update_email(reply["original_mail_id"], {"state": "replied"})


if __name__ == "__main__":
    penpal = AiPenpal()
    while True:
        penpal.check_new_messages()
        penpal.wait()
