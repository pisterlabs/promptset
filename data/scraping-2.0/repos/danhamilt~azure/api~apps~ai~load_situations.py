

test_situations = [
    "I'm not earning as much as my peers. I'm a failure.",
    "I broke a glass. I'm so clumsy.",
    "I got lost even with GPS. I'm hopeless.",
    "I overslept and missed my class. I'm irresponsible.",
    "I said something awkward. Everyone must think I'm weird.",
    "I didn't help when someone dropped their things. I'm rude.",
    "I didn't make a good impression. They won't like me.",
    "I tripped on the sidewalk. I'm always making a fool of myself.",
    "I didn't get the joke everyone laughed at. I'm out of place.",
    "I forgot my friend's baby's name. I'm thoughtless.",
    "I spent too much money on clothes. I'm wasteful.",
    "I couldn't lift the heavy box. I'm weak.",
    "I got nervous and blanked during my presentation. I'm a loser.",
    "I didn't get a callback from the job interview. I'm not good enough.",
    "I'm not as fit as I used to be. I'm letting myself go.",
    "I snapped at my partner over something small. I'm too sensitive.",
    "I forgot to send a thank you note. I'm ungrateful.",
    "I didn't notice my friend got a haircut. I'm self-centered.",
    "I'm not able to save money. I'll never have a secure future.",
    "I made a mistake at work. Everyone will think I'm incompetent."
    "I got a negative review at work. I'm not good at anything.",
    "I forgot to send an important email. I'm so disorganized.",
    "I argued with my partner again. I'm ruining this relationship.",
    "I didn't clean the house. I'm such a slob.",
    "I overreacted to a joke. I'm too sensitive.",
    "I didn't get selected for the team. I lack talent.",
    "I burnt the toast again. I can't do the simplest things right.",
    "I missed the bus. I can't manage my time.",
    "I didn’t make it to the gym today. I lack dedication.",
    "I lost my temper with my friend. I'm a bad friend.",
    "I didn’t understand the material. I'm not smart enough.",
    "I made a mistake in my finances. I'm irresponsible.",
    "I didn’t stand up for what I believed in. I'm a pushover.",
    "I yelled at my kids. I'm a terrible parent.",
    "I didn’t get the job offer. I'm a failure.",
    "I forgot to return a borrowed item. I'm thoughtless.",
    "I didn’t help my colleague when they were struggling. I'm selfish.",
    "I missed my doctor’s appointment. I'm negligent.",
    "I didn’t perform well in the game. I'm a disappointment.",
    "I spent too much money on shopping. I'm reckless.",
    "I forgot my anniversary. I'm a terrible spouse.",
    "I didn’t back up my files and lost data. I'm careless.",
    "I didn’t follow through on my promise. I'm unreliable.",
    "I said something insensitive. I'm a terrible person.",
    "I can’t stick to a routine. I'm a failure."
]
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.document_loaders import JSONLoader
from apps.ai.models import ChatSituation
import dj_database_url
from django.conf import settings
import json
COLLECTION_NAME = 'chat_situations'
CONNECTION_STRING = 'postgresql://postgres:postgres@localhost:5432/postgres'
def get_cosmos_db():
    cdb = settings.DATABASES['cosmos']
    return f'postgresql://{cdb["USER"]}:{cdb["PASSWORD"]}@{cdb["HOST"]}:5432/{cdb["NAME"]}'

def create_json_file():
    situations = [{
        'text': cs.text,
        'predicted_emotions': json.loads(cs.predicted_emotions) \
            if isinstance(cs.predicted_emotions, str) else cs.predicted_emotions
    } for cs in ChatSituation.objects.only('text','predicted_emotions')]
    with open('situations.json', 'w') as f:
        json.dump(situations, f)

def load_test_situations():
    with open('/Users/danielhamilton/Downloads/tokens.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.lower().startswith('person:'):
            start, situation = line.split(': ')
            ChatSituation.objects.create(text=situation.strip())


def get_vector_store():
    model: str = "text-davinci-002"
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment=model, chunk_size=1)
    index_name: str = "langchain-vector-demo"
    vector_store_address: str = 'https://django.search.windows.net'
    vector_store_password: str = '3FFRo3KzjIXHQPfGuFlVQLKX9PdTAmq1nKxkDWQVlMAzSeD83H2L'
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )
    return vector_store

def load_old_situations():
    with open('/Users/danielhamilton/azure/situations.json', 'r') as f:
        situations = json.load(f)
    for situation in situations:
        try:
            ChatSituation.objects.create(text=situation['text'], predicted_emotions=situation['predicted_emotions'])
        except:
            pass

def main():
    file_path = '/Users/danielhamilton/azure/situations.json'
    loader = JSONLoader(file_path=file_path, 
                        jq_schema='.[]',
                        content_key='text')
    data = loader.load()
    vector_store = get_vector_store()
    vector_store.add_documents(data)
    with open('/Users/danielhamilton/azure/final_output.txt', 'w+') as f:
        for ts in test_situations:
            f.write(f'\nSituation {ts}\n')
            docs = vector_store.similarity_search(
                query=ts,
                k=3,
                search_type='similarity',
            )
            vs_sit = docs[0].page_content
            f.write(f"Vector search similar: {docs[0].page_content}\n")
            try:
                vs_cs = ChatSituation.objects.get(text=vs_sit)
                f.write(f"Vector search emotions: {vs_cs.predicted_emotions} \n")
            except:
                f.write("could not find in database")

            cs, _ = ChatSituation.objects.get_or_create(text=ts)
            f.write(json.dumps(cs.predicted_emotions))