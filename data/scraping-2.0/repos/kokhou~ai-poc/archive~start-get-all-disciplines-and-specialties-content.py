import os

import openai
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
# from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY"),
# specialties = [
#     "Cardiology"
# ]
specialties = [
    "Cardiology", "Dermatology", "Gastroenterology", "Neurology", "Orthopedics",
    "Pediatrics", "Ophthalmology", "Urology", "Pulmonology", "Rheumatology",
    "Endocrinology", "Obstetrics", "Gynecology", "Nephrology", "Hematology",
    "Otolaryngology", "Infectious Disease", "Allergy and Immunology", "Psychiatry",
    "Radiology", "Anesthesiology", "Oncology", "Plastic Surgery", "Physical Therapy",
    "Geriatrics", "Family Medicine", "Internal Medicine", "General Surgery",
    "Cardiothoracic Surgery", "Vascular Surgery", "Neonatology", "Sports Medicine",
    "Pain Management", "Podiatry", "Dental", "Geriatric Medicine", "Neonatal-Perinatal Medicine",
    "Reproductive Endocrinology", "Transplant Surgery", "Bariatric Surgery",
    "Colorectal Surgery", "Gastrointestinal Surgery", "Maxillofacial Surgery",
    "Forensic Medicine", "Hospice and Palliative Medicine", "Interventional Radiology",
    "Pediatric Surgery", "Nuclear Medicine", "Sleep Medicine", "Medical Genetics"
]

for i, specialty in enumerate(specialties, start=1):
    with get_openai_callback() as cb:
        pre_prompt = """[INST] <<SYS>>\n
        you are a professional in the medical field, 
        I will provide a healthcare disciplines and specialties and you will explain what is the healthcare disciplines and specialties for, 
        explain in deep and step by step, give 10 most common possibilities use case, but not-creative and not-fake. 
        I will use this for embedding to vector db to answer questions about medical field.
        give me long answer, at least 2000 words,
        
        \n\n"""
        context = "CONTEXT:\n\n{context}\n" + "Question: {question}" + "[\INST]"
        prompt = PromptTemplate(template=pre_prompt + context, input_variables=["context", "question"])
        llm_chain = LLMChain(prompt=prompt, llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5))
        result = llm_chain.run(context="", question=f"{specialty}")

        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

        # Open the file in write mode ('w')
        with open(f'specialty/{specialty}.txt', 'w') as file:
            # Write the text to the file
            file.write(result)
