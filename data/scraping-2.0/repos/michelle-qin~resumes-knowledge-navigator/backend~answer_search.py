import fitz
import os
from openai import AzureOpenAI
import ast
import json
from sql_helpers import evaluate_query, evaluate_query_blind, get_text_from_id
from openai_helper import get_client

"""
This function will search the text for the answer to a given question.

param: doc_id (int) the unique id allowing us to find the processed text and pdf filename in the file_table->documents sql table
param: question (string) the question we are trying to answer.
return: dictionary with two fields
    1. Answer which is the answer to the question
    2. Citation which is the verbatim text supporting it

"""
def search_text(doc_id, query):
    document = get_text_from_id(doc_id)

    search_prompt = f"""
    You are acting as an agent that will search through a document for the answer to a request. I will now give you the document.
    Document: "{document}"
    Now, I will give you the request.
    Request: "{query}"
    Given the passage and the request, you must give the verbatim citation from the given passage which satisfies the request. If the information is not explicitly shown in the text just put "None". Make sure your answer is in this format:
    {{
        "answer": "<YOUR ANSWER>",
        "citation": "<YOUR CITATION>",
    }}

    I will now give you an example so that you can learn how to do this task. If you are given the following document:
    Document: "ADULT EDUCATION INSTRUCTOR
    Experience
    Company Name City , State Adult Education Instructor 08/2016 to Current Developed a diploma program that fit the needs of the community,
    continues to work with the community and wants to see the students succeed move on into either industry or collegeÂ
    Company Name City , State Agriculture/Credit Recovery Teacher 08/2000 to Current
    Planned and conducted activities for a balanced program of instruction, demonstration, and work time that provided students with
    opportunities to observe, question, and investigate.
    Goal Setting Established clear objectives for all lessons/projects and communicated with students, achieving a total understanding of grading
    rubric and overall class expectations."
    and you are given the following request:
    Request: "What was the title of their most recent job?"
    Then, your answer should be:
    {{
        "answer": "Adult Education Instructor",
        "citation": "Company Name City , State Adult Education Instructor 08/2016 to Current Developed a diploma program that fit the needs of the community,
    continues to work with the community and wants to see the students succeed move on into either industry or collegeÂ"
    }}

    Here's another example:
    Request: "Show me their accounting experience."
    Then, your answer should be:
    {{
        "answer": "None",
        "citation": "None">
    }}

    Only give the answer in the format that I told you. Do not say anything else extra other than the answer. Do not act as if you are a human. Act as if you are the raw output for a query. Give only the first instance of the answer even if multiple parts are relevant
    """
    client = get_client()
    response = client.chat.completions.create(
        model = "gpt4",
        temperature = 0,
        messages=[
            {"role": "system", "content": "Assistant is acting as an agent that will search through a document for the answer to a request."},
            {"role": "user", "content": search_prompt}
        ]
    )
    response = response.choices[0].message.content

    try:
        json_dict = json.loads(response)
    except:
        raise ValueError("The LLM outputted a non-json-formattable string. Contact Thomas/Daniel to work this out.")

    return json_dict

"""
This code will add a highlight to a pdf given a piece of text that the LLM has searched for.

param: input_path (string) the path to the pdf file we will be highlighting
param: output_path (string) the path that we want to save the highlighted pdf to
param: sections (List[string]) the list of text sections we want to highlight in the pdf
"""
def add_hyperlinks_to_pdf(input_path, output_path, sections):
    pdf_document = fitz.open(input_path)
    for query in sections:
        for page in pdf_document:
            search_results = page.search_for(query)
            for rect in search_results:
                annot = page.add_highlight_annot(rect)
    pdf_document.save(output_path)
    pdf_document.close()


def query_gpt4(prompt):
    client = get_client()
    response = client.chat.completions.create(
        model="gpt4",
        messages=[
            {"role": "user", "content": prompt}
            ])
    return response.choices[0].message.content


def multiple_document_table(doc_ids, query):

    client = get_client()
    schema = "multiple_doc"
    table_name = "table"

    field_prompt = f"""
    Given the query, give me the name of the column that would store the answer to it in a SQL table. Here are a few examples:

    Query: Show me how this applicant has demostrates diversity.
    Field name: diversity

    Query: What foreign experience does this applicant have?
    Field name: foreign_experience

    Query: What college did they go to?
    Field name: college

    Remember only give me the field name after the "Field name:" This should be one word with no spaces. Use an underscore to separate words.

    Query: {query}
    Field name:
    """

    field = query_gpt4(field_prompt)

    res = {}

    res["doc_id"] = []
    res[field] = []
    res[f"{field}_citation"] = []


    for doc_id in doc_ids:

        response_dict = search_text(doc_id, query)

        res["doc_id"].append(doc_id)
        res[field].append(response_dict["answer"])
        res[f"{field}_citation"].append(response_dict["citation"])

    return res
            

def multiple_document_table_to_sql(doc_ids, query):

    client = get_client()
    schema = "multiple_doc"
    table_name = "table"

    field_prompt = f"""
    Given the query, give me the name of the column that would store the answer to it in a SQL table. Here are a few examples:

    Query: Show me how this applicant has demostrates diversity.
    Field name: diversity

    Query: What foreign experience does this applicant have?
    Field name: foreign_experience

    Query: What college did they go to?
    Field name: college

    Remember only give me the field name after the "Field name:" This should be one word with no spaces. Use an underscore to separate words.

    Query: {query}
    Field name:
    """

    field = query_gpt4(field_prompt)

    delete_query = f"DROP TABLE search_results"
    evaluate_query(schema, delete_query)

    create_query = f"""
        CREATE TABLE search_results (
        doc_id INTEGER,
        {field} TEXT,
        {field}_citation TEXT
    );
    """
    print(create_query)
    evaluate_query(schema, create_query)

    for doc_id in doc_ids:
        print(f"Processing document {doc_id}")
        # print(get_text_from_id(doc_id))

        response_dict = search_text(doc_id, query)

        insert_query = f"""
        INSERT INTO search_results (doc_id, {field}, {field}_citation)
        VALUES (?, ?, ?);
        """

        data = (
            doc_id,
            response_dict["answer"],
            response_dict["citation"]
        )

        print(insert_query)
        evaluate_query_blind(schema, insert_query, data)
            
