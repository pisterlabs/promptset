import openai
import pandas as pd
import os

openai.api_key = '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'

# Load the patient data
def load_patient_data(filename):
    data = pd.read_csv(filename)
    patient_data = {}
    for patient_name in data['Name'].unique():
        patient_records = data[data['Name'] == patient_name]
        phq9_records = patient_records[patient_records['Test'] == 'PHQ-9'][['Date', 'Score']].to_dict('records')
        gad7_records = patient_records[patient_records['Test'] == 'GAD-7'][['Date', 'Score']].to_dict('records')
        patient_data[patient_name] = {
            "PHQ-9": {"scores": phq9_records},
            "GAD-7": {"scores": gad7_records}
        }
    return patient_data

def generate_response(patient_name, patient_data):
    patient = patient_data[patient_name]
    
    phq9_scores = patient["PHQ-9"]["scores"]
    gad7_scores = patient["GAD-7"]["scores"]
    
    latest_phq9_score = phq9_scores[-1]["Score"]
    latest_gad7_score = gad7_scores[-1]["Score"]
    
    phq9_trend = latest_phq9_score - phq9_scores[0]["Score"]
    gad7_trend = latest_gad7_score - gad7_scores[0]["Score"]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that helps patients analyze their health."},
        {"role": "assistant", "content": "Hello! I'm your patient assistant. Can I have your name please?"},
        {"role": "user", "content": f"My name is {patient_name}."}
    ]

    messages.append({"role": "assistant", "content": f"Thank you, {patient_name}. Your latest PHQ-9 score is {latest_phq9_score} and your latest GAD-7 score is {latest_gad7_score}. The trend in your PHQ-9 scores is {phq9_trend} and the trend in your GAD-7 scores is {gad7_trend}. How are you feeling today?"})
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    
    messages.append({"role": "assistant", "content": response['choices'][0]['message']['content'].strip()})
    
    messages.append({"role": "user", "content": "Can you analyze my data and recommend an action for me?"})
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    
    return response['choices'][0]['message']['content'].strip()


def main():
    patient_data = load_patient_data("patient_data.csv")
    
    print("Hello! I'm your patient assistant. Can I have your name please?")
    patient_name = input()

    while patient_name.lower() != 'quit':
        if patient_name in patient_data:
            print(generate_response(patient_name, patient_data))
            print("Do you have another question or concern?")
        else:
            print("I'm sorry, but I couldn't find that patient in the database. Please enter a valid patient name.")

        patient_name = input()

if __name__ == "__main__":
    main()
