import json

import pandas as pd
from langchain import PromptTemplate, OpenAI
import os

os.environ['OPENAI_API_KEY'] = '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'


def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def generate_response(context_string):
    """
    Generates a response to a given context string using OpenAI's text-davinci-003 model.

    Args:
        context_string (str): A string containing context information about a website page.

    Returns:
        str: A concise response in at most two sentences that examines the context, communicates the type of website,
        provides summaries of website topics and its overarching purpose, elucidates the pursued goals, primary focus,
        and the values emphasized for stakeholders.
    """

    template = """I gathered the context information about a website page.

    Context:
    {context}
    
    Examine the context, communicate the type of website, provide summaries of website topics and its overarching 
    purpose, elucidate the pursued goals, primary focus, and the values emphasized for stakeholders. Use natural 
    language and be as descriptive as possible in your analysis. Provide a concise response in at most two sentences.
    
    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["context"],
        template=template
    )

    openai = OpenAI(
        model_name="text-davinci-003",
    )

    return openai(prompt_template.format(context=context_string))


def process_reports_and_save_csv(report_paths, output_csv):
    """
    Processes a list of report files in JSON format, generates a response for each report's context, and saves the results
    to a CSV file.

    Args:
        report_paths (list): A list of file paths to report files in JSON format.
        output_csv (str): The file path to save the results to as a CSV file.

    Returns:
        None
    """
    results = []

    for report_path in report_paths:
        json_data = load_json_data(report_path)
        context = json_data['context']

        response = generate_response(context)
        response_text = response.strip()

        results.append({
            'Report File': report_path,
            'Context': context,
            'Response': response_text
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    report_files = ["../data/report6.json"]
    output_csv_file = "../out/output_results.csv"

    process_reports_and_save_csv(report_files, output_csv_file)
