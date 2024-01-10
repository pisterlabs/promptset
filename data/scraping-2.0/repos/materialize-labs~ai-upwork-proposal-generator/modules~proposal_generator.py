import sqlite3
import json
import openai
import time
from modules.database import insert_fine_tuned_model, insert_model_response

def fetch_job_application_data():
    conn = sqlite3.connect('applications.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
    SELECT jobs.job_category_level_one, jobs.job_category_level_two,
           jobs.description, jobs.engagement, applications.cover_letter
    FROM jobs
    JOIN applications ON jobs.id = applications.job_id
    LIMIT 10;
    """
    cur.execute(query)
    records = cur.fetchall()
    conn.close()

    return records

def prepare_training_data(records):
    # Convert records to a list of dictionaries
    records_list = [dict(record) for record in records]

    # Convert records into required training format
    formatted_records = []
    for record in records_list:
        prompt_input = f'Job Category Level One: {record["job_category_level_one"]}\n'
        prompt_input += f'Job Category Level Two: {record["job_category_level_two"]}\n'
        prompt_input += f'Description: {record["description"]}\n'
        prompt_input += f'Engagement: {record["engagement"]}\n\n###\n\n'

        formatted_records.append({"prompt": prompt_input, "completion": f' {record["cover_letter"]} END'})

    # Convert list of formatted records into JSONL
    jsonl_data = "\n".join(json.dumps(formatted_record) for formatted_record in formatted_records)
    
    return jsonl_data

def save_training_data_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        f.write(data)

def upload_training_data(file_path):
    with open(file_path, "rb") as f:
        response = openai.File.create(purpose="fine-tune", file=f)
    return response["id"]

def create_fine_tuned_model(training_file_id, model="davinci"):
    result = openai.FineTune.create(
        training_file=training_file_id,
        model=model)

    # Insert the newly created fine tuned model details into the database
    insert_fine_tuned_model(result)

    return result

def list_fine_tuned_models():
    return openai.FineTune.list()

def delete_fine_tuned_model(model_id):
    result = openai.Model.delete(model_id)
    return result

def generate_completions(model, prompt, max_tokens, stop, n, temperature):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        stop=stop,
        n=n,
        temperature=temperature
    )

    # Save the response to a JSON file
    with open('completions.json', 'w') as f:
        json.dump(response, f, ensure_ascii=False, indent=4)

    insert_model_response(prompt, model, max_tokens, stop, n, temperature, response)

    return response