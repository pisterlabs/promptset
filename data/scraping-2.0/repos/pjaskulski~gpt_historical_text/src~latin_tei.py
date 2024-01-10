""" openai test - extraction info about parents, children, wife,
    husband from bio
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import openai


env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

data = "Francisco Cruswicz magistro in theologia [in civitate Constantiensi commo-ranti]: Eidem decano eccl. s. Floriani extra muros Crac. (15 m. arg. p.), canonica-tus in eccl. s. Michaelis in castro Crac. (12 m. arg. p.), per obitum apud SA ea va-cante Nicolai Falkemberg vacans, confertur.;s. m. scholastico et cantori ac custodi Crac. XI, XIII."
response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Tag text and convert to TEI XML format:\n\n {data}",
    temperature=0.3,
    max_tokens=500,
    top_p=1.0,
    frequency_penalty=0.8,
    presence_penalty=0.0
)

print(response['choices'][0]['text'])

file_output = Path("..") / "output" / "latin_tei.txt"
with open(file_output, 'w', encoding='utf-8') as f:
    f.write(response['choices'][0]['text'])