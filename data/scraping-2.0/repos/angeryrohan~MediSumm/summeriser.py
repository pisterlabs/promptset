import os
import openai
from dotenv import load_dotenv
import itertools
import threading
import time
import sys
from recent_info import *
load_dotenv()
my_api_key = os.environ.get('API_KEY')
openai.api_key = my_api_key


def problems_summary(my_arr):
    if(my_arr != [('','')]):
        problems_msg=""
        for problem, recent_date in my_arr:
            problems_msg+= " Problem: " + problem + "." + " date: " + recent_date + "."
        return problems_msg
    else:
        return ''

def medications_and_directions_summary(my_arr):
    if(my_arr != [('','','')]):
        mnd_msg=""
        for med,direc,my_date in my_arr:
            mnd_msg+= " Medication: " + med + ". Direction: " + direc + ". Date: " + my_date
        return mnd_msg
    else:
        return ''

def reason_for_visit_summary(my_arr):
    if(my_arr != [('','')]):
        rfv_msg=""
        for rfv, date in my_arr:
            rfv_msg+= " Reason for visit: " + rfv + "." + " date: " + date + "."
        return rfv_msg
    else:
        return ''

def treatment_plan_summary(my_arr):
    if(my_arr != [('','')]):
        tp_msg=""
        for tp, date in my_arr:
            tp_msg+= " Treatment Plan: " + tp + "." + " date: " + date + "."
        return tp_msg
    else:
        return ''

def encounter_diagnosis_summary(my_arr):
    if(my_arr != [('','')]):
        ed_msg=""
        for ed, date in my_arr:
            ed_msg+= " Encountered Diagnosis: " + ed + "." + " date: " + date + "."
        return ed_msg 
    else:
        return ''

def allergies_summary(my_arr):
    if(my_arr != [('','')]):
        for (allergen, severity) in my_arr:
            allergy_msg =  "Allergy: " + allergen + "." + " Severity: " + severity + "."
            return allergy_msg
    else:
        return ''

# function to create an array  
def compose_message(dim):
    final_message = ""
    final_message += problems_summary(get_recent_problems(dim, filled_information)) + medications_and_directions_summary(get_recent_medication(dim, filled_information)) + reason_for_visit_summary(get_recent_reason_for_visit(dim, filled_information)) + treatment_plan_summary(get_recent_treatment_plan(dim,filled_information)) + encounter_diagnosis_summary(get_recent_EncounterDiagnosis(dim,filled_information)) + allergies_summary(get_recent_allergies(filled_information))
    return "Give me a summary about a patient in about " + str(dim*100) +" words without loosing any major health information. Don't mention dates too much, but highlight the patterns in them. Here is the information about the patient: " + final_message

done = False
#here is the animation
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\r Making a concise and smart summary. Please Wait ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\Almost Done, Loading Web Page!     ')

t = threading.Thread(target=animate)
t.start()

input_oms =  compose_message(1)
output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": 
    input_oms}]
    )
oms = str(output['choices'][0]['message']['content'])


input_tms = compose_message(2)
output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": 
      input_tms}]
    )
tms = str(output['choices'][0]['message']['content'])


input_fms = compose_message(5)
output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": 
      input_fms}]
    )
fms = str(output['choices'][0]['message']['content'])


done = True




