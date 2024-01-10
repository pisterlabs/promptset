import openai
import os
import time

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

def gpt4_classify_email(sms):
    gotAnswer = False
    while gotAnswer == False: 
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a system that classifies messages as either being spam or not spam",
                    },
                    {"role": "user", "content": sms},
                ],
            )
            gotAnswer = True
            
            classification = response["choices"][0]["message"]["content"].strip()
            if classification == "spam":
                return "spam"
            else:
                return "not spam"
        except:
            print("Error, trying again") 
            time.sleep(0.005)

comments = [
    'Hey John, just wanted to check in on you to see how your doing. Hows the move been?',
	'This is an important message from Wells Fargo. Please sign in here to review a suspicious login to your account: https://bit.ly/12345678',
	'This is the FBI. If you do not click this link you will be ARRESTED and put IN JAIL.',
	'We have your granddaughter. We will hurt her unless you pay 28.485 ETH to the address 0xb0d042f70c02630ea30975017e03cb50140f594afc0af717413a12f0a56e3174 in the next 24 hours.',
	'This is the Federal Investigations Department. IRS records show that there are a number of overseas transactionsunder your name, you need to pay the full portion of the transaction fees to the IRS Department, which you never did. You currently owe $5,638.38. Please call us as soon as possible at (415)-555-3437.'
]

# print(gpt4_classify_email(comments[1]))