import asyncio
import ssl
import csv
import json
import openai
import os
from dotenv import load_dotenv


class GET_OUTCOME:
    
    @staticmethod
    def remove_empty_lines(content):
        lines = content.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def short_completion(prompt):
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[
                    {"role": "system", "content": "You are a pre-processing expert, cleaning up noisy and redundant data when data is swapped from the website."},
                    {"role": "user", "content": 'Based on the data provided below because I swapped data from the application\'s privacy policy link. Eliminate noise and redundant words.\n\n' + prompt}
                ]
            )
            assistant_reply = response.choices[0].message['content']
            return assistant_reply
        except Exception as e:
            return "Error: The prompt length exceeds the maximum allowed length of 8192 tokens."
        

    @staticmethod
    async def loop_csv(input_csv_path, output_csv_path):
        with open(input_csv_path, "r", newline="", encoding="utf-8") as csvfile, \
            open(output_csv_path, "w", newline="", encoding="utf-8") as outputfile:
            
            reader = csv.reader(csvfile)
            writer = csv.writer(outputfile)

            headers = next(reader)
            writer.writerow(headers)

            for index, row in enumerate(reader):
                print("\n_____________ Run times " +
                    row[0] + " <" + row[2] + "> " + "_____________")
                
                privacy_policy_content_short = GET_OUTCOME().short_completion(row[7])

                row[headers.index("privacy_policy_short")] = GET_OUTCOME().remove_empty_lines(privacy_policy_content_short)
                
                writer.writerow(row)
                
                print("~~~~~~~~~~~~~~ Success ~~~~~~~~~~~~~~\n")




async def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    input_csv_path = "/Users/nghiempt/Observation/sr-dps/sr-dps-server/bonus_v3/filtered_app_privacy_policy_part2.csv"
    output_csv_path = "/Users/nghiempt/Observation/sr-dps/sr-dps-server/bonus_v3/filtered_app_privacy_policy2.csv"

    await GET_OUTCOME().loop_csv(input_csv_path, output_csv_path)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
