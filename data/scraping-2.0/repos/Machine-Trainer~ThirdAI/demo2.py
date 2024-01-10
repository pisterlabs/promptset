from thirdai import licensing
licensing.activate("389C76-A6CCC7-BA2CAE-02A467-313084-V3")
from thirdai import neural_db as ndb
from utils import CSV
import openai
checkpoint = "qna_db_1"
db = ndb.NeuralDB(user_id="my_user")
import os
if not os.path.exists(checkpoint):
    os.system("wget -O qna_db_1.zip 'https://www.dropbox.com/scl/fi/s1zhxmwjpayj5jphzct0p/qna_1_db.zip?dl=0&rlkey=ftcgrzt1rpc2d6hx0iuk1lz1r'")
    os.system("unzip qna_db_1.zip -d qna_db_1")

db.from_checkpoint("qna_db_1")
csv_files = ['UpdatedResumeDataSet.csv' ]
csv_docs = []

for file in csv_files:
    csv_doc = CSV(
        path=file,
        id_column='DOC_ID',
        strong_columns=['Resume'],
        weak_columns=['Category'],  
        reference_columns=['Resume'])

    csv_docs.append(csv_doc)
source_ids = db.insert(csv_docs, train=False)

# TODO: Add find keywords related to JobDescription.
search_results = db.search(
    query="Java Developer",
    top_k=1,
    on_error=lambda error_msg: print(f"Error! {error_msg}"))


# Set up your OpenAI API credentials
openai.api_key = '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'

# Define a list of messages to start the conversation
messages = [
    {'role': 'system', 'content': 'You act as a professional Software Engineer resume expert, you help tailor my resume and give out the tailored resume.'},
    {'role': 'user', 'content': 'I will provide you with some resume example related to this position: '},
]

for result in search_results:
    messages.append({'role': 'user', 'content': result.text()})
    print('************')
messages.append({'role': 'system', 'content': 'Please provide me with your resume.'})
# Extract the text from my resume.
file_path = 'Weijian Zeng_SDE.txt'  # Replace with the actual file path
encoding = 'latin-1'
with open(file_path, 'r', encoding=encoding) as file:
    content = file.read()
# Close the file
file.close()

messages.append({'role': 'user', 'content': 'Here is my resume:'+content})

# Make a request to ChatGPT
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
)

# Get the assistant's reply
reply = response['choices'][0]['message']['content']
print(reply)

file_path = 'result.txt'  # Replace with the desired file path

with open(file_path, 'w') as file:
    file.write(reply)
print("Results saved successfully.")