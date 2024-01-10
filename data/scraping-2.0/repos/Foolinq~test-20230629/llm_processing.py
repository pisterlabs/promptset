import os
from langchain import LLMChain, OpenAI, PromptTemplate

def process_llm_request(doctor_question, patient_name, patient_age, patient_weight, patient_height, patient_bmi, patient_sex, patient_activity, symptoms):
    api_key = os.getenv('OPENAI_API_KEY')
    llm_request = f"{doctor_question} The patient's name is {{patient_name}}, age is {{patient_age}}, weight is {{patient_weight}}, height is {{patient_height}}, BMI is {{patient_bmi}}, sex at birth is {{patient_sex}}, and recent physical activity level is {{patient_activity}}. The reported symptoms are {{symptoms}}."
    llm = LLMChain(llm=OpenAI(api_key=api_key), prompt=PromptTemplate(input_variables=["patient_name", "patient_age", "patient_weight", "patient_height", "patient_bmi", "patient_sex", "patient_activity", "symptoms"], template=llm_request))
    # Call the function to process the request with the LLM model
    result = llm.predict(patient_name=patient_name, patient_age=patient_age, patient_weight=patient_weight, patient_height=patient_height, patient_bmi=patient_bmi, patient_sex=patient_sex, patient_activity=patient_activity, symptoms=symptoms)
    return result
