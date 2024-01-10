#!/usr/bin/env python3.9
import argparse
import openai

def main():
    parser = argparse.ArgumentParser(description='Chat parser para el chat de pepper')
    parser.add_argument('-m','--message', help='Message to be sent for ChatGPT', required=True)

    msg = parser.parse_args()['message']

    openai.api_key = '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'
    model="gpt-3.5-turbo"

    chat = msg

    messages = [{"role": "system", "content": "You are a robot called maqui, your model is Pepper from Softbank robotics. You give short answers of no more than 50 words."},
                {"role": "user", "content": chat}]
    reply = openai.ChatCompletion.create(model=model, messages=messages)

    return reply["choices"][0]["message"]["content"]

if __name__ == '__main__':
    main()
