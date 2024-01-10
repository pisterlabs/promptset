import openai
import csv
import re
import time

def generate_answer(text, api_key):
    """ 使用GPT-3.5生成答案 """
    openai.api_key = api_key
    prompt = f"Extract open triples (subject, relation, object) from the following text: \"{text}\""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except openai.error.RateLimitError:
        wait_seconds = 20
        print(f"Rate limit reached, waiting for {wait_seconds} seconds...")
        time.sleep(wait_seconds)
        return generate_answer(text, api_key)  # Retry

def process_csv(input_csv_file, output_csv_file, api_key):
    """ 处理CSV文件，并使用GPT-3.5进行关系抽取 """
    with open(input_csv_file, mode='r', encoding='utf-8-sig') as infile, open(output_csv_file, mode='a', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ["Extracted Entity1", "Extracted Entity2", "Extracted Non-Entity Content", "Original Text"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        if outfile.tell() == 0:
            writer.writeheader()

        for row in reader:
            text = row['文本']
            print(f"Processing: {text}")
            extracted_info = generate_answer(text, api_key)
            parts = extracted_info.split(', ')
            entity1, relation, entity2 = parts[:3] if len(parts) >= 3 else ("N/A", "N/A", "N/A")
            writer.writerow({"Extracted Entity1": entity1, "Extracted Entity2": entity2, "Extracted Non-Entity Content": relation, "Original Text": text})
            print(f"Processed: {text}")
            time.sleep(10)


process_csv("input_csv_path", "output_csv_path", "api_key")