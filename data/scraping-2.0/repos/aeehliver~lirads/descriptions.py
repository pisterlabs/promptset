import os
import openai
import jsonlines
import sys
import re
import csv
import pandas as pd
from fpdf import FPDF

# Inicializar OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]

# Specify how many questions you want to generate for each description
num_generated_questions = 100  # <-- You can change this number according to your needs

# Global variables
tokens_consumed = 0
total_cost_accumulated = 0

def display_tokens_and_cost(tokens_for_current_request):
    global tokens_consumed
    global total_cost_accumulated
    
    tokens_consumed += tokens_for_current_request
    input_cost_per_token = 0.03 / 1000
    output_cost_per_token = 0.06 / 1000
    
    current_request_cost = (tokens_for_current_request / 2) * (input_cost_per_token + output_cost_per_token)
    total_cost_accumulated += current_request_cost
    
    sys.stdout.write("\r" + " " * 100 + "\r")  # Clear line
    sys.stdout.write(f"Processing {category}. Tokens this request: {tokens_for_current_request}. Total tokens: {tokens_consumed}. Current cost: ${current_request_cost:.4f}. Total cost: ${total_cost_accumulated:.4f}")
    sys.stdout.flush()

def generate_gpt_based_qna_with_token_display(category, description, examples):
    dataset_entries = []

    # Generate additional examples based on the provided examples
    for example in examples:
        clean_example = re.sub(r"^\d+\.\s*", "", example)
        try:
            messages = [
                {"role": "system", "content": "You are going to help me generate medical records based on exact examples. You should follow the same structure and content as the example and use the description provided to help you contextualize. Do not add unrelated information and do not deviate from the style of the example."},
                {"role": "user", "content": f"Using the following clinical history format: {clean_example}. And with the following reference description: {description}. Generate a new clinical history that follows the same format and style but with different, extremely creative problems and with complex medical issues at times and simple at others that give the same result. Do it without the writing exceeding 100 words."}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )
            
            dataset_entries.append({"text": response.choices[0]['message']['content'].strip(), "label": category})

            # Viewing tokens and costs
            display_tokens_and_cost(response['usage']['total_tokens'])

        except Exception as e:
            with open('errors.txt', 'a', encoding='utf-8') as err_file:
                err_file.write(f"Error processing example for {category}: {e}\n")

    return dataset_entries

dir_path = r'd:\bot\liverai\lirads'
lirads_categories = ["LR-1", "LR-2", "LR-3", "LR-4", "LR-5", "LR-M", "LR-NC", "LR-TIV"]
final_dataset = []

for category in lirads_categories:
    try:
        with open(os.path.join(dir_path, f"{category}.txt"), 'r', encoding='utf-8') as desc_file:
            description = desc_file.read()

        with open(os.path.join(dir_path, f"{category}e.txt"), 'r', encoding='utf-8') as ex_file:
            examples = ex_file.readlines()
        
        final_dataset.extend(generate_gpt_based_qna_with_token_display(category, description, examples))

    except FileNotFoundError:
        with open('errors.txt', 'a', encoding='utf-8') as err_file:
            err_file.write(f"Error: No se pudo encontrar el archivo {category}.txt o {category}e.txt.\n")

# Save the data set to a jsonl, csv and pdf file
with jsonlines.open(os.path.join(dir_path, 'lirads_dataset_final.jsonl'), 'w') as json_file:
    json_file.write_all(final_dataset)

df = pd.DataFrame(final_dataset)
df.to_csv(os.path.join(dir_path, 'lirads_dataset_final.csv'), index=False)

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Generated LIRADS Dataset', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin = 15)
pdf.set_font("Arial", size = 12)

for row in final_dataset:
    pdf.multi_cell(0, 10, f"Text: {row['text']}\nLabel: {row['label']}\n\n")

pdf.output(os.path.join(dir_path, 'lirads_dataset_final.pdf'))

def calculate_and_display_total_cost():
    input_cost_per_token = 0.03 / 1000
    output_cost_per_token = 0.06 / 1000
    total_cost = (tokens_consumed / 2) * (input_cost_per_token + output_cost_per_token)
    
    print(f"\n\n----- Resumen Final -----")
    print(f"Total tokens consumed: {tokens_consumed}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Dataset saved in: {os.path.join(dir_path, 'lirads_dataset_final.jsonl')}, .csv y .pdf")
    print(f"-------------------------")

calculate_and_display_total_cost()
