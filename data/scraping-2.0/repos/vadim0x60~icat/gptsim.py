from google.cloud import bigquery
import openai
import random
import re

# Initialize the BigQuery client
client = bigquery.Client(project='graphsim')

admissions = list(client.query('SELECT DISTINCT hadm_id FROM `graphsim.mimic.allevents`'))

while True:
    prompt_adm, test_adm = random.sample(admissions, 2)
    prompt = []
    # Print the results
    prompt.append(f'===== PATIENT {prompt_adm[0]} =====')
    for row in client.query('SELECT label, intensity FROM `graphsim.mimic.allevents` WHERE hadm_id = ' + str(prompt_adm[0]) + ' ORDER BY eventtime'):
        prompt.append('\t'.join(str(x) for x in row if x))
    prompt.append(f'===== PATIENT {test_adm[0]} =====')
    i = False
    for row in client.query('SELECT label, intensity FROM `graphsim.mimic.allevents` WHERE hadm_id = ' + str(test_adm[0]) + ' ORDER BY eventtime'):
        prompt.append('\t'.join(str(x) for x in row if x))
        
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Predict the next line in the hospital log file. Don't say anything else"},
                {"role": "user", "content": '\n'.join(prompt[:500])}
            ],
            temperature=0.0,
            stop='\n'
        )
        predicted_line = completion.choices[0].message['content']
        print(predicted_line)
        labelmatch = re.fullmatch(r'(.+)\s+([0-9]+\.[0-9]+)', predicted_line)
        if labelmatch:
            label_pred = labelmatch.group(1)
            intensity_pred = labelmatch.group(2)
        else:
            label_pred = predicted_line
            intensity_pred = 'NULL'

        label_true = row.label
        intensity_true = row.intensity if row.intensity else 'NULL'

        client.query(f'INSERT INTO `graphsim.mimic.results` (model, hadm_id, label_true, label_pred, intensity_true, intensity_pred) VALUES ("gpt-4", {test_adm[0]}, "{label_true}", "{label_pred}", {intensity_true}, {intensity_pred})').result()