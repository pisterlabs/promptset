from typing import List, Tuple

import json
import openai
import os
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from datetime import datetime
import locale

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

locale.setlocale(locale.LC_TIME, "it_IT.UTF-8")

def extract_fhir_objects(text: str) -> Tuple[List[dict], List[dict]]:
    # This function will take a string as input and return two lists of FHIR objects
    # For the sake of this example, it will just return two empty lists
    # You will need to implement the logic for extracting the MedicationRequest and Appointment objects from the text

    current_date = datetime.now().strftime("%d %B %Y")
    template_string = """
Il testo fornito tra triple backticks (```) contiene istruzioni per l'applicazione di un farmaco\n
e per l'organizzazione di una visita di follow-up. Pertanto, la risposta che mi aspetto avrà esclusivamente\n
due liste di oggetti JSON corrispondenti: una per `MedicationRequest` FHIR e una per `Appointment` FHIR.\n
La risposta deve essere in puro JSON, un oggetto con due proprietà: "medicationRequests" e "appointments",\n
che sono liste contenenti gli oggetti corrispondenti. Per quanto riguarda medicationRequests, concentrati\n
ad estrarre dosageInstruction e timing. Per quanto riguarda Appointment, quando ambiguo, considera le date\n
rispetto alla data odierna.\n
```${prescrizione}```
"""

    prompt_template = ChatPromptTemplate.from_template(template_string)
    
    llm = ChatOpenAI(temperature=0.0, model_name='gpt-4')
    chain = LLMChain(llm=llm, prompt=prompt_template)

    result = chain(text)
    inner_json = result['text']
    inner_dict = json.loads(inner_json)
    medication_requests = inner_dict['medicationRequests']
    appointments = inner_dict['appointments']
    return medication_requests, appointments