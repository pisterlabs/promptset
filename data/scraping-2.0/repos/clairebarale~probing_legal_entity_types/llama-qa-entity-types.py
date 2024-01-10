from transformers import AutoTokenizer, pipeline
import transformers
import torch
import argparse
import string
from tqdm import tqdm
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import pandas as pd
import os

# list of entity types
list_entity_types_long = ["city", "country", "region", "state", "province", "area", "nation", "land", "republic", "district", "territory", "division", "zone","date", "day_of_the_month", 
                          "appointment", "particular date", "date stamp", "time", "timestamp", "calendar date", "schedule""nationality", "religious community", "political group", "ethnic groups", 
                          "community", "racial group", "party", "faction", "ideological group", "belief community","tribunal", "firm", "ngo", "company", "corporation", "business", 
                          "nonprofit", "association", "charity", "court", "judicial body", "convention", "international convention", "law", "legislation", "legal code", "treaty", 
                          "agreement", "protocol", "statute","plausibility", "authenticity", "integrity", "trustworthiness", "reliability", "credibility", "believability", "credibility", 
                          "credibleness", "verdict", "result", "resolution", "judgment", "approval", "denial", "decline", "rejection", "approval", "determination", "finding", "conclusion", 
                          "decision", "grant", "refusal", "positive decision", "negative decision", "data", "employment", "resident", "national", "inhabitant", "information", "gender", "age", 
                          "citizen", "citizenship", "sex", "job", "occupation", "profession", "affidavit", "documentary evidence", "proof", "testimony", "exhibit", "record", "file", "paperwork", 
                          "operation", "procedure", "legal procedure", "legal process", "judicial procedure", "legal steps", "judicial process", "proof", "evidence", "document", "written document", 
                          "written evidence", "written proof", "written record", "written report", "written statement", "written testimony", "written witness statement", "explanation", 
                          "clarification", "interpretation", "reason", "ground", "legal ground", "justification", "rationale", "foundation", "legal basis", "legal justification", "citation", 
                          "jurisprudence", "case", "law", "case law", "Legal case", "lawsuit", "Legal matter", "legal precedent", "judicial decisions", "legal rulings", "country report", "report", 
                          "official report", "written report", "ngo report", "national report", "state report", "regional report", "nonprofit report", "non-governmental organization report", 
                          "charity report"]

list_entity_types_short = ["location","date","nationality","community","group","tribunal","company","convention","law","credibility","judgment","determination","employment",
                           "gender","age","citizenship","procedure","evidence","explanation","reason","legal ground","case law","legal precedent","country report"]


parser = argparse.ArgumentParser()   
# ARGUMENTS
parser.add_argument('--in_file', dest='in_file', action='store', required=True, help='the path of the input file')
parser.add_argument('--device', dest='device', action='store', required=True, help='cpu or cuda')
args = parser.parse_args()
args = vars(args)
in_file = args['in_file'] 
device = args['device']


# reads the input file, assuming a tab-separated format, and extracts the entities from each line
entities = [] # my list of entities
with open(in_file, 'r') as _:
    lines = [line.replace('\n', '') for line in _.readlines()]
        
    for line in lines:
        line = line.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        line = line.strip() # remove whitespace at the beginning and end of the line
        entities.append(line.split('\t')[0])
        # Now "entities" is a list contraining the entities extracted from the dataset, single and multi token
    entities.pop(0) # remove the header
    entities = entities[:10000]

    # measure the length of the entities in token
    entities_token_length = [len(entity.split()) for entity in entities]


    # count the number of single token entities, 2 tokens, 3 tokens and 3+ tokens
    single_token_entities = 0
    two_token_entities = 0
    three_token_entities = 0
    three_plus_token_entities = 0
    for length in entities_token_length:
        if length == 1:
            single_token_entities += 1
        elif length == 2:
            two_token_entities += 1
        elif length == 3:
            three_token_entities += 1
        else:
            three_plus_token_entities += 1
    print(f"single_token_entities: {single_token_entities}")
    print(f"two_token_entities: {two_token_entities}")
    print(f"three_token_entities: {three_token_entities}")
    print(f"three_plus_token_entities: {three_plus_token_entities}")

# MODEL
model = "meta-llama/Llama-2-7b-chat-hf"

#tokenizer = AutoTokenizer.from_pretrained(model)

#PIPELINE
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
llm = HuggingFacePipeline(pipeline = pipeline)

template = """This is an entity typing task. If you don't know the answer, just say that you don't know, don't try to make up an answer. Answer with the correct entity type only. There is a list of entity types: {list_entity_types_short}.
Question: What entity type is {entity}?
Answer:"""

# define the prompts
predictions = []
for e in tqdm(entities):
    list_prompts = []
    prompt = PromptTemplate.from_template(template)
    chain= prompt | llm
    entity=e
    result = chain.invoke({"list_entity_types_short": list_entity_types_long, "entity": e})
    predictions.append(result)

# df and write to csv
df = pd.DataFrame()
df["entity"] = entities
df["prediction"] = predictions
print(df)
folder = "llama2_qa_entity_types/"
entity_in_file = in_file.split("/")[-1].split(".")[0]
output_filename = f"output_{entity_in_file}.csv"
output_file = os.path.join(folder, output_filename)
print(output_file)
df.to_csv(output_file, sep=";", index=False)


