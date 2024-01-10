#!/usr/bin/env python3

import sys
import openai

def main():
    # Load your API key from an environment variable or secret management service
    openai.api_key = "{KEYPLACEHOLDER}"
    if len(sys.argv) > 1:
        question = sys.argv[1]
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpfull assistant."},
                        {"role": "user", "content": question},
                    ]
            )
        gpt = response['choices'][0]['message']['content']
        print(gpt)
    else:
        print("No question was provided.")

if __name__ == '__main__':
    main()
  