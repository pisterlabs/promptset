import weaviate
import openai
import os
import json
import PyPDF2
import re
from math import ceil

"""
Denne koden leser pdf-er fra mappen pdf_data. Der ligger personalhåndboken.
For øyeblikket brukes kun clusters fra weaviate med varighet 14 dager. Må derfor lage nytt clusters etter det. Man kan bruke denne koden på ferskt cluster
Leser over pdf-ene og lager litt mindre biter av det. Bruker embedding til open ai for å finne en plass til det i databasen.
Er ikke ment å kjøre i "prod", men fint å ha med resten av repoet.
"""

openai_key = os.getenv('OPENAI_KEY')
weaviate_key = os.getenv('WEAVIATE_KEY')
weaviate_url = os.getenv('WEAVIATE_URL')
openai.api_key = os.getenv('OPENAI_KEY')

pdf_dir = 'pdf_data'

pdf_data = {}

def clean(string):
    # Remove newline characters
    string = string.replace('\n', ' ')

    # Remove URLs
    string = re.sub(r'http\S+', '', string)

    # Remove Unicode literals
    #string = re.sub(r'U+2022', '', string)

    #Remove stopwords
    stopwords = ["simployer", "published", "last", "updated", "print", "date", "paper", "version", "will", "not", "be", "maintained", "Current", "version", "will", "always", "be", "available", "on", "the", "applications", "website", "current"]
    string = string.split()
    filtered_words = [word for word in string if word.lower() not in stopwords]
    string = " ".join(filtered_words)

    return string


# iterate over each PDF file in the folder
for filename in os.listdir(pdf_dir):
    if filename.endswith('.pdf'):
        # open the PDF file in binary mode
        with open(os.path.join(pdf_dir, filename), 'rb') as f:
            # initialize a PdfFileReader object to read the PDF file
            pdf_reader = PyPDF2.PdfReader(f)
            # extract the text from each page of the PDF file
            for p in range(len(pdf_reader.pages)):
                name_and_page = filename+str(p)
                text = pdf_reader.pages[p].extract_text()
                text = clean(text)
                if len(text)>250:
                    group_size = 250
                    overlap_size = 50
                    new_paras = [text[i:i+group_size] for i in range(0, len(text), group_size-overlap_size)]
                    for n in range(len(new_paras)):
                        name_page_and_part = name_and_page +"part"+ str(n)
                        pdf_data[name_page_and_part] = new_paras[n]

# write the PDF data to a JSON file
with open('output.json', 'w') as f:
    json.dump(pdf_data, f)

openai.api_key = os.environ["OPENAI_KEY"]

client = weaviate.Client(
    url=weaviate_test_url,
    auth_client_secret=weaviate.auth.AuthApiKey(api_key=weaviate_test_key),
    additional_headers={
        "X-OpenAI-Api-Key": openai_key
    }
)

#Nytt schema
schema = {
  "classes": [
    {
      "class": "Personalbok",
      "description": "Innhold fra personalhåndboken til Norconsult.",
      "vectorizer": "text2vec-openai",
      "moduleConfig": {
        "text2vec-openai": {
          "model": "ada",
          "modelVersion": "002",
          "type": "text"
        }
      },
      "properties": [
        {
          "dataType": [
            "text"
          ],
          "description": "Avsnitt fra personalhåndboken",
          "moduleConfig": {
            "text2vec-openai": {
              "skip": "false",
              "vectorizePropertyName": "false"
            }
          },
          "name": "content"
        }
      ]
    }
  ]
}

#Lage nytt schema
client.schema.create(schema)
#client.schema.delete_all()

#Her får vi inn tekst av personalhåndboka

new_data = []
personalhandbok = 'output.json'
with open(personalhandbok) as file:
    data = json.load(file)
    for v in data.values():
        new_data.append(v)

#Unngå å kjøre mer enn en gang per oppstart
client.batch.configure(
  batch_size=5,
)

with client.batch as batch:
    for s in new_data:
        batch.add_data_object(
          data_object={"content": s},
          class_name="Personalbok"
        )