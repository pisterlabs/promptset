import asyncio
import ssl
from urllib.request import urlopen
from bs4 import BeautifulSoup
import csv
import json
import openai
import os
from dotenv import load_dotenv
from newspaper import Article


class READ_PRIVACY_POLICY:

    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    def get_completion(prompt, model="gpt-4"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.9,
        )
        return response.choices[0].message.content

    @staticmethod
    def remove_empty_lines(content):
        lines = content.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(cleaned_lines)

    @staticmethod
    def check_valid_token_prompt(prompt):
        return len(prompt) <= 8000
        
    @staticmethod
    def generate_result(url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = READ_PRIVACY_POLICY.remove_empty_lines(article.text)
            
            # response = openai.ChatCompletion.create(
            #     model='gpt-4',
            #     messages=[
            #         {"role": "system", "content": "You are a pre-processing expert, cleaning up noisy and redundant data when data is swapped from the website."},
            #         {"role": "user", "content": 'Based on the data provided below because I swapped data from the application\'s privacy policy link. Eliminate noise and redundant words.\n\n' + text}
            #     ]
            # )
            
            # assistant_reply = response.choices[0].message['content']
            
            return text
        except Exception as e:
            print(f"An exception occurred: {e}")
            return "An error occurred during processing"


class GET_OUTCOME:

    @staticmethod
    async def loop_csv(input_csv_path, output_csv_path, read_privacy_policy):
        with open(input_csv_path, "r", newline="", encoding="utf-8") as csvfile, \
            open(output_csv_path, "w", newline="", encoding="utf-8") as outputfile:
            
            reader = csv.reader(csvfile)
            writer = csv.writer(outputfile)

            # Write the header to the output CSV
            headers = next(reader)
            writer.writerow(headers)

            for index, row in enumerate(reader):
                print("\n_____________ Run times " +
                    row[0] + " <" + row[2] + "> " + "_____________")
                
                privacy_policy_content = read_privacy_policy.generate_result(row[6])

                row[headers.index("privacy_policy_raw")] = privacy_policy_content
                
                print("Privacy Policy: " + privacy_policy_content)

                writer.writerow(row)
                
                print("~~~~~~~~~~~~~~ Success ~~~~~~~~~~~~~~\n")




async def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
        
    read_privacy_policy = READ_PRIVACY_POLICY()

    input_csv_path = "/Users/nghiempt/Observation/sr-dps/sr-dps-server/bonus_v2/app_privacy_policy.csv"
    output_csv_path = "/Users/nghiempt/Observation/sr-dps/sr-dps-server/bonus_v2/app_privacy_policy2.csv"

    await GET_OUTCOME().loop_csv(input_csv_path, output_csv_path, read_privacy_policy)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())