#!/usr/bin/env python3

import os
import sys
import random
import openai
import requests

"""
This script will trigger an email to a receipient of your choice with a random reason for having your Travel Trailer
on your own property since HOA's love to flex.

Reqired env vars:

OPENAI_API_KEY
MAILGUN_API_KEY
MAILGUN_DOMAIN
MAILGUN_RECEIPIENT
MAILGUN_SENDER
"""


class EmailQuery:
    """
    a class for generating and sending emails
    """

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.mg_api_key = os.getenv("MAILGUN_API_KEY")
        self.mg_domain = os.getenv("MAILGUN_DOMAIN")
        self.mail_receipient = os.getenv("MAILGUN_RECEIPIENT")
        self.mg_sender = os.getenv("MAILGUN_SENDER")
        self.reasons = ["maintenance of roof", "preparing for camping trip", "fixing slide problem", "electrical issue"]

    def emailQuery(self):
        """
        write an email to send to my HOA based on some made-up reason
        :return:
        """
        reason = random.choice(self.reasons)
        print("Generating email response with {} as an excuse...".format(reason))

        response = openai.Completion.create(model="text-davinci-003",
                                            prompt="Write my HOA a letter explaining my travel trailer will be at my house for a few days because of {}".format(
                                                reason),
                                            temperature=0.7,
                                            max_tokens=100,
                                            top_p=1.0,
                                            frequency_penalty=0.0,
                                            presence_penalty=0.0)

        if 'choices' not in response:
            print("unable to get response from chatgpt: {}".format(response))
            sys.exit(1)
        else:
            if len(response['choices']) > 0:
                answer = response['choices'][0]['text']
            else:
                print("unable to get response from chatgpt: {}".format(response))
                sys.exit(1)
        return answer

    def sendEmail(self, answer=None):
        # Your Mailgun domain
        domain = self.mg_domain
        # The recipient of the email
        to = self.mail_receipient
        # The sender of the email
        from_email = self.mg_sender
        # The subject of the email
        subject = 'Hello, from OpenAI'
        # The body of the email
        text = answer
        # The data to send in the POST request to the Mailgun API
        data = {
            'from': from_email,
            'to': to,
            'subject': subject,
            'text': text
        }

        # Send the POST request to the Mailgun API
        response = requests.post(
            f'https://api.mailgun.net/v3/{domain}/messages',
            auth=('api', self.mg_api_key),
            data=data
        )

        # Print the response from the Mailgun API
        print(response.text)


def main():
    eq = EmailQuery()
    ans = eq.emailQuery()
    if len(ans) > 0:
        print("Sending email...")
        eq.sendEmail(answer=ans)
    print("finished, answer was: ")
    print(ans)


if __name__ == "__main__":
    main()
