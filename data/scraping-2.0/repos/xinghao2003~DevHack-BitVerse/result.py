from config import client, temp_path
from langchain.document_loaders import UnstructuredFileLoader
import json

result_candidate = []

def load_system_prompt():
    loader = UnstructuredFileLoader("./prompts/result_system_2.txt")
    docs = loader.load()
    content = docs[0].page_content
    return content


def result(parsed_job_details, parsed_candidates, process):
    potential_candidate = generate_result(parsed_job_details, parsed_candidates)
    if(potential_candidate is not None):
        for id in potential_candidate['candidates_id']:
            for candidate in parsed_candidates:
                if (candidate['id'] == id):
                    result_candidate.append(candidate)

    process.update(label="Completed", expanded=False, state="complete")
    return result_candidate


def generate_result(parsed_job_details, parsed_candidates):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "system", "content": f"{load_system_prompt()}"},
                  {"role": "user", "content": f"{parsed_job_details}, {parsed_candidates}"}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    print(response)

    if (response.choices[0].message.content == None):
        return None

    response_json = json.loads(
        response.choices[0].message.content)
    
    return response_json
