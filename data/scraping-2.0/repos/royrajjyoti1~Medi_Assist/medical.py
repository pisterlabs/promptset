from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import re
import os 
from dotenv import load_dotenv
import openai

load_dotenv()
 
endpoint = os.getenv("ENDPOINT")
key = os.getenv("KEY")
model_id = os.getenv("MODEL_ID")




def output_fun(formUrl):
    document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key))
    
    poller = document_analysis_client.begin_analyze_document_from_url(model_id, formUrl)
    result = poller.result()
    #print("result",result)
    return extract_from_content_v2(result)
  
def extract_from_content_v2(result):
    formatted_strings = []
    for idx, document in enumerate(result.documents):
        # Extract fields from document
        fields = document.fields
        # Extract and format basic patient details
        p_name = fields.get('p_name', {}).value if fields.get('p_name', {}).value else ''
        p_age = fields.get('p_age', {}).value if fields.get('p_age', {}).value else ''
        p_gender_value = fields.get('p_gender', {}).value if fields.get('p_gender', {}).value else ''
        # Map gender value to full form
        gender_map = {'M': 'Male', 'F': 'Female', 'O': 'Other'}
        p_gender = gender_map.get(p_gender_value, 'Unknown')
        # Append patient details to the formatted strings list
        formatted_strings.append(f"Patient Name: {p_name}")
        formatted_strings.append(f"Patient Age: {p_age}")
        formatted_strings.append(f"Patient Gender: {p_gender}")
        formatted_strings.append('-' * 40)  # Separator for better readability
        # Extract and format CBC details
        cbc_fields = fields.get('CBC', {}).value if fields.get('CBC', {}).value else []
        for entry in cbc_fields:
            try:
                tests_value = entry.value['tests'].value
                result_value = entry.value['result'].value
                units_value = entry.value['units'].value
                formatted_strings.append(f"Test: {tests_value}, Result: {result_value}, Units: {units_value}")
            except:
                #print('Data Format Is Not Supported')
                continue
        # Add a separator after each patient's details
        formatted_strings.append('=' * 60)
    # Concatenate the formatted strings into a single string
    final_string_v2 = "\n".join(formatted_strings)
    #print(final_string_v2)
    return final_string_v2

def remark(final_output_v6):
    openai.api_key = os.getenv('OPEN_AI_KEY')
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": "Generate remark in bullet points " + final_output_v6
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    return response['choices'][0]['message']['content']

def remark_hindi(remark_result):
    openai.api_key = os.getenv('OPEN_AI_KEY')
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": "Translate to hindi " + remark_result
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    return response['choices'][0]['message']['content']

def recommend_Medication(final_output_v6):
    openai.api_key = os.getenv('OPEN_AI_KEY')
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": "Generate Medicines/Tablets and medicine composition which will help to treat patient of condition " + final_output_v6
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    return response['choices'][0]['message']['content']
    
    
def recommend_Medication_hindi(recommend_Medication_result):
    openai.api_key = os.getenv('OPEN_AI_KEY')
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": "Translate to hindi" + recommend_Medication_result
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    return response['choices'][0]['message']['content']
    