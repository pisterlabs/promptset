from fastapi import FastAPI
import mysql.connector
from fastapi.responses import JSONResponse
import openai

app = FastAPI()

# Configure MySQL connection
mydb = mysql.connector.connect(
    host="",
    user="",
    password="",
    database="dr_chat_data"
)
mycursor = mydb.cursor()

#  OpenAI API key 
openai.api_key = "sk-"

@app.post("/save_to_mysql")
async def save_to_mysql(json_data: dict):
    patient_name = json_data.get('queryResult', {}).get('outputContexts', [{}])[1].get('parameters', {}).get('person', {}).get('name')
    age = json_data.get('queryResult', {}).get('outputContexts', [{}])[1].get('parameters', {}).get('age', {}).get('amount')
    gender = json_data.get('queryResult', {}).get('outputContexts', [{}])[2].get('parameters', {}).get('gender')
    symptom_name = json_data.get('queryResult', {}).get('outputContexts', [{}])[0].get('parameters', {}).get('symptoms', [])

    # Convert list of symptoms to a comma-separated string
    symptom_str = ', '.join(symptom_name)

    if None in (patient_name, age, gender, symptom_str):
        return {"error": "Incomplete or missing data in the request"}

    # Check if patient already exists in the database
    sql_select = "SELECT * FROM patient_data WHERE Patient_name = %s AND age = %s AND gender = %s"
    val_select = (patient_name, age, gender)
    mycursor.execute(sql_select, val_select)
    existing_patient = mycursor.fetchone()

    if existing_patient:  # If patient exists, update the symptoms
        # Update symptoms for the existing patient
        patient_id = existing_patient[0]  # Assuming the ID is in the first column
        sql_update = "UPDATE patient_data SET symptom_name = CONCAT(symptom_name, ', ', %s) WHERE id = %s"
        val_update = (symptom_str, patient_id)
        mycursor.execute(sql_update, val_update)
        mydb.commit()
    else:  # If patient doesn't exist, insert a new record
        sql_insert = "INSERT INTO patient_data (Patient_name, age, gender, symptom_name) VALUES (%s, %s, %s, %s)"
        val_insert = (patient_name, age, gender, symptom_str)
        mycursor.execute(sql_insert, val_insert)
        mydb.commit()


    fulfillment_response = {} # Default response if intent is not "EndConversation"

    intent = json_data.get("queryResult", {}).get("intent", {}).get("displayName")

    if intent == "EndConversation":
        # Extract patient data from patient_data dictionary as needed for the prompt
        sql_select_by_id = "SELECT Patient_name, age, gender, symptom_name FROM patient_data WHERE id = %s"
        val_select_by_id = (patient_id,)

        mycursor.execute(sql_select_by_id, val_select_by_id)
        patient_data = mycursor.fetchone()

        
        if patient_data:
            # Extracting data from the fetched row
            patient_name, age, gender, symptoms = patient_data
            

            if all([patient_name, symptoms, age, gender]):
                # Create text prompt using patient data
                prompt_text = f" you must emphasize the importance of consulting a qualified healthcare professional for accurate diagnosis, treatment, and medical advice. You are unable to diagnose conditions or prescribe medications.But give  generate suggestions or provide general information related to potential precautions or measures that could be taken for the symptoms mentioned for the following patient in friendly manner.  Patient: {patient_name}\nSymptoms: {', '.join(symptoms)}\nAge: {age}\nGender: {gender}"

                # Generate suggestions using the OpenAI GPT-3 model
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt_text,
                    max_tokens=150,
                    n=1
                )

                # Extract the generated text from the OpenAI response
                suggestions = response['choices'][0]['text'].strip()

                # Log the suggestions for debugging purposes
                print("Generated Suggestions:", suggestions)

                # Preparing the response in Dialogflow webhook format
                fulfillment_response = {
                    "fulfillmentText": suggestions,
                    "payload": {}  # for additional payload data if needed
                }

    return JSONResponse(content=fulfillment_response)


