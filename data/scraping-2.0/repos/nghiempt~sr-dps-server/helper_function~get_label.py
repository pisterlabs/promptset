import asyncio
import ssl
import csv
import json
import openai
import os
from dotenv import load_dotenv


class CHATGPT_GENERATOR:

    @staticmethod
    def get_prompt(ds_content, pp_content):
        prompt = """Let's compare and analyze the information between Data Safety and Privacy Policy to clarify 3 issues: which information is incorrect, which information is incomplete and which information is inconsistent. Notes when classifying: Incomplete: Data Safety provides information but is not as complete as the Privacy Policy provides. Incorrect: Data Safety does not provide that information, but the Privacy Policy mentions it. Inconsistency: Data Safety is provided but its description is inconsistent with the Privacy Policy information provided. Note: always gives me the result (0 or 1, 1 is yes, 0 is no) in the form below: {"label" : { "incorrect": (0 or 1),  "incomplete": (0 or 1), "inconsistent": (0 or 1) }, "label_description" " { "incorrect": "explaination", "incomplete": "explaination", "inconsistent": "explaination" } } . Please in the answer, just give me the json only and in English. Below is information for 2 parts: Data Safety: """ + ds_content + """, Privacy Policy: """ + pp_content
        return prompt
    
    @staticmethod
    def get_completion(prompt):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.chat.completions.create(
            model=os.getenv("GPT_4"),
            messages=[
                {"role": "system", "content": "You are an assistant who analyzes and evaluates the correct, complete, and consistency between the Data Safety information provided compared to the information provided by the Privacy Policy of applications on the Google Play Store."},
                {"role": "user", "content": prompt}
            ]
        )
        assistant_reply = response.choices[0].message.content
        return assistant_reply
    

class DATASET_GENERATOR:

    @staticmethod
    def loop_csv(input_csv_path, output_csv_path, chatgpt_generator):
        with open(input_csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            new_data = []
            for index, row in enumerate(reader):
                try: 
                    print("\n_____________ Run times App ID: " + row['app_id'] + "_____________")
                    prompt = chatgpt_generator.get_prompt(row['app_data_safety'], row['app_privacy_policy'])
                    result = chatgpt_generator.get_completion(prompt)
                    new_row = [row['app_id'], result]
                    new_data.append(new_row)
                except Exception as e:
                    print(e)
                    new_row = [row['app_id'], "ERROR"]
                    new_data.append(new_row)
                    print("~~~~~~~~~~~~~~ ERROR ~~~~~~~~~~~~~~\n")
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['app_id', 'label_content'])
            csv_writer.writerows(new_data)


if __name__ == "__main__":
    chatgpt_generator = CHATGPT_GENERATOR()

    input_csv_path = "../dataset/formated_data/dataset.csv"
    output_csv_path = "../dataset/formated_data/label.csv"


    DATASET_GENERATOR().loop_csv(input_csv_path, output_csv_path, chatgpt_generator)
