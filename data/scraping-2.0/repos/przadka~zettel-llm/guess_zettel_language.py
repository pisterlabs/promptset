import os
import openai
import pandas as pd
import time
import ast


openai.api_key = os.getenv("OPENAI_API_KEY")

REQUESTS_PER_MINUTE = 1000
SLEEP_INTERVAL = 60.0 / REQUESTS_PER_MINUTE

SYSTEM_MSG = '''
You are a helpful, respectful and honest assistant. 
===
'''

def guess_language(quote):
    """Guess language of a quote."""
    
    # Construct the user message
    user_msg = f'''
Please guess the language of the following text:
===
{quote}
===

Output only the language code, e.g. "en" for English, "de" for German, "fr" for French, etc.
'''

    print(f"System message: {SYSTEM_MSG}")
    print(f"User message: {user_msg}")

    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=30,
            temperature=0.0,
        )
        
        return response['choices'][0]['message']['content'].strip()

    except openai.error.InvalidRequestError as e:
        print(f"Error while processing the request: {e}")
        return "ERROR_TRIGGERED_CONTENT_MANAGEMENT_POLICY"


def main():
    """Main function to process queries."""

    df = pd.read_csv("documents/fine_tuning.csv")

    df["Lang"] = ""

    for index, row in df.iterrows():
        print(f"Processing row {index}...\n")
        quote = row["quotation"]
        lang = guess_language(quote)
        print(f"Guessed language: {lang}")
        df.at[index, "Lang"] = lang
        print("\n")

        # Sleep for the calculated interval to respect the rate limit
        time.sleep(SLEEP_INTERVAL)

    # Save the dataframe with guessed language back to a new CSV
    df.to_csv("documents/fine_tuning.csv", index=False)

if __name__ == "__main__":
    main()