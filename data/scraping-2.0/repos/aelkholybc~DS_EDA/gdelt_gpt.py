import psycopg2
from tqdm import tqdm
import openai

conn = psycopg2.connect(database="coredata",
                host="postgresql-general-test.cff6npe4bzmg.us-west-2.rds.amazonaws.com",
                user="coredata",
                password="5cPMhnSS27sFKgCFRvXxKpc9e",
                port="5432")

# query = """
# with n as (
# select "date","RECORDID","DOCUMENTIDENTIFIER" as source_url,lat,lon from gdelt_base_terms
# where "date"::date>date('2023-11-10')
# ),
# o as (
# select n.*,b.text
# from n inner join gdelt_embeddings b
# on n."RECORDID"=b.recordid
# where b.text ilike '%outbreak%'
# and (
#     b.text ilike  '%severe%'
#     OR b.text ilike '%significant'
#     OR b.text ilike '%large%')
# )
# select * from o
# where 
# o.text ilike '%pneumonia%'
# OR o.text ilike '%respiratory%'
# """


query = """
SELECT gdelt_base_terms."date", gdelt_base_terms."RECORDID", gdelt_base_terms."DOCUMENTIDENTIFIER" as source_url, gdelt_base_terms.lat, gdelt_base_terms.lon, gdelt_embeddings.text
FROM gdelt_base_terms
INNER JOIN gdelt_embeddings ON gdelt_base_terms."RECORDID" = gdelt_embeddings.recordid
WHERE 
    gdelt_base_terms."date" BETWEEN '2021-08-25' AND '2021-12-01' AND
    (
        to_tsvector('english', gdelt_embeddings.text) @@ to_tsquery('english', 'shortage | rising & cost')
    ) AND
    (
        to_tsvector('english', gdelt_embeddings.text) @@ to_tsquery('english', 'hurricane & ida | storm & ida')
    );
"""

with conn.cursor() as cur:
    cur.execute(query)
    rows = cur.fetchall()

model = 'gpt-4-32k'
openai.api_type = "azure"
openai.api_key = '7bcf1af3b56c49fa816f22daeb51a021'
openai.api_base = "https://brainchainuseast2.openai.azure.com"
openai.api_version = "2023-08-01-preview"


import code; code.interact(local=dict(globals(), **locals()))

MAX_RETRIES = 5
out = []
for row in tqdm(rows):
    text = row[5]
    content = f"""
    Take a deep breath, and read the following instructions carefully.
    1. Go through the attached article text and pick out all products, goods, or commodities that are mentioned. These are tangible goods and NOT services/utilities (e.g. not electricity or labor).
    2. If one of them is mentioned as being in short supply or having a spike in demand/price, write down the following information about it as a JSON:
        a. product: What product is in short supply?
        b. reason: What is the reason for the shortage?
        c. duration: How long is the shortage expected to last?
        d. location: Where is the shortage happening?
        e. impact: What is the impact of the shortage?
    (If you are not able to find anything useful, just return the json with blank entries for each of the above keys)
    Make SURE each entry refers ONLY to a product that people buy or sell. For example, a spike in pneumonia is NOT what you should look for, but a shortage of vaccines IS what you want.
    DOUBLE CHECK that the product mentioned is, in fact, experiencing some kind of shortage or surge in demand. The idea is the product you find should be something people are talking about as being scarce, hard to find, or anticipated to be in short supply.
    For each CORRECT entry you find, you recieve $1,000. For each INCORRECT entry you find, you lose $100,000. If you aren't able to do this work, every puppy in heaven cries in sadness.
    ARTICLE:
    {text}
    """
    messages=[
        {"role": "system", "content": "You are an expert analyst. You are looking for supply chain disruptions. Return only JSON"},
        {"role": "user", "content": content}
    ]
    retries = 0
    while retries < MAX_RETRIES:
        try:
            res = openai.ChatCompletion.create(
                engine=model,
                messages=messages,
            )
            print(res['choices'][0]['message']['content'])
            out.append({"id": row[1], "date": row[0], "lat": row[3], "lon": row[4], "text": row[5], "json": res['choices'][0]['message']['content']})
            if res['choices'][0]['message']['content']:
                break
        except Exception as e:
            retries += 1
            print("retrying: {}".format(str(e)))
        if retries == MAX_RETRIES:
            print('error')

import code; code.interact(local=dict(globals(), **locals()))