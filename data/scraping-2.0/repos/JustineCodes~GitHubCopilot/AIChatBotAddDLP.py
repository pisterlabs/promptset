# Create chatbot that will take in user input, ask ChatGPT, and return the response.

import openai, json, secrets, my_secrets
import os
from nightfall import Confidence, DetectionRule, Detector, RedactionConfig, MaskConfig, Nightfall

openai.api_key = my_secrets.OPENAI_API_KEY

with open('api_key.txt', 'r') as f:
    api_key = f.read().strip()

nightfall = Nightfall(api_key)



def chat_with_gpt(message):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=message,
        max_tokens=100,
        temperature=0.6,
        n=1,
        stop=None,
        timeout=15,
        )
    reply = response.choices[0].text.strip()
    return reply


while True:
    user_input = input("You: ")
    payload = [user_input]

# Define an inline detection rule that looks for Likely Credit Card Numbers and redacts them
    detection_rule = DetectionRule([
                            Detector(
                                min_confidence=Confidence.LIKELY,
                                nightfall_detector="CREDIT_CARD_NUMBER",
                                display_name="Credit Card Number",
                                redaction_config=RedactionConfig(
                                    remove_finding=False, 
                                    substitution_phrase="[REDACTED]")
                            )])

    findings, redacted_payload = nightfall.scan_text(
                        payload,
                        detection_rules=[detection_rule])
    
    if redacted_payload[0]:
        message_body = redacted_payload[0]
    else:
        message_body = payload[0]

    print("After content filtering - this is what will be sent to ChatGPT:\n\n", message_body, "\n\n----\n\n")

    if user_input.lower() in ['quite', 'exit']:
        print("ChatGPT: Goodbye!")
        break
    else:
        response = chat_with_gpt(message_body)
        print("ChatGPT: " + response)
