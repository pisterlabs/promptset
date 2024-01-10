import json
import openai

# Load API key from json file

with open('tokens/openaikey.json', 'r') as keyf:
    openai.api_key = json.load(keyf)['key']


def openai_request(body_text, intro):
    # Create request to API
    response = openai.Completion.create(model="text-davinci-003", prompt=intro + body_text + " ```",
                                        temperature=0, max_tokens=2000)
    return response['choices'][0]['text']


# specify comments file
with open('E:/Research/Data_pubmed/processed_data_authors2/cognitive_authors_u_aff.csv', 'r', encoding="utf8") as csvfile:
    csvtext = csvfile.readlines()


list_affiliations = []
for line in csvtext:
    list_affiliations.append(tuple(line.strip().split(', ')))

# roll back to working request?
prompt_intro = '''[Task: Departmental Text Classification to CSV] 

Categories: [Biology, Psychology, Biotechnology&Genetics, Medical, Health Sciences, Pathology&Pharmacology, Neuroscience, Engineering&Informatics, Chemistry&Physics&Math]

Output Format: "Id, Category" 

Do not add header to the output.
Text to classify: '''

# previous: Output Format: "Id, Affiliation, Category"

output = 'Id, Category'
# 50 affiliations ~ 2,6k tokens, splitting by 25 records
step = 100
pos = 300500

while pos+step < len(list_affiliations):
    batch = str([" ".join(list_affiliations[i]).replace('\'', '').replace('\"', '') for i in range(pos, pos+step)])
    processed = openai_request(batch, prompt_intro)
    output += '\n' + processed
    pos += step

    if divmod(pos, 500)[1] == 0: # check if reminder is equal to zero (every thousand records)
        with open('E:/Research/Data_pubmed/processed_data_authors2/cognitive_lbld/cognitive_lbld_' + str(pos) + '.csv',
                  'w', encoding="utf-8") as f:
            f.write(output)

        print("Processed records: ", pos)

        output = 'Id, Category'

with open('E:/Research/Data_pubmed/processed_data_authors2/cognitive_lbld/cognitive_lbld_' + str(pos) + '.csv',
          'w', encoding="utf-8") as f:
    f.write(output)
