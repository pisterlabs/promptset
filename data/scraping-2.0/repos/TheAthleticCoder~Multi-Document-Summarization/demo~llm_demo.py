import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from torch import cuda
import torch
from tqdm import tqdm
import pandas as pd
import time
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
import csv

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def cleanData(sentence):
    return stemmer.stem(sentence)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("\n Starting the MMR Process \n\n")

def calculateSimilarity(sentence, doc):
    if doc == []:
        return 0
    vocab = {}
    # For each word in the sentence, add it to the vocabulary dictionary
    for word in sentence:
        vocab[word] = 0
    # Initialize an empty string to hold the document in one sentence
    docInOneSentence = ''
    # For each term in the document, add it to the docInOneSentence string
    # and add each word in the term to the vocabulary dictionary
    for t in doc:
        docInOneSentence += (t + ' ')
        for word in t.split():
            vocab[word]=0
    # Initialize a CountVectorizer with the vocabulary dictionary as the vocabulary
    cv = CountVectorizer(vocabulary=vocab.keys())
    # Fit transform the document into a vector
    docVector = cv.fit_transform([docInOneSentence])
    sentenceVector = cv.fit_transform([sentence])
    return cosine_similarity(docVector, sentenceVector)[0][0]
def concat(x):
    x = ' '.join(x)
    x = x.split('\n')
    # Filter out any strings in the list that are just a space
    x = list(filter(lambda s: not s == ' ', x))
    # Remove leading and trailing whitespace from each string in the list
    x = list(map(lambda s: s.strip(), x))
    return x

def get_sentences(texts, sentences, clean, originalSentenceOf):
    # Split the text into sentences
    parts = texts.split('.')
    
    for part in parts:
        cl = cleanData(part)
        
        sentences.append(part)
        clean.append(cl)
        
        # Map the cleaned part to the original part in the originalSentenceOf dictionary
        originalSentenceOf[cl] = part
    
    # Remove duplicates from the clean list by converting it to a set
    setClean = set(clean)

    return setClean

import signal

# Define a handler function that raises an exception when called
def handler(signum, frame):
    raise Exception("Function execution took too long")
signal.signal(signal.SIGALRM, handler)
import operator

def get_mmr(doc, alpha):
    try:
        # Set an alarm for 60 seconds
        signal.alarm(60)
        
        sentences = []
        clean = []
        originalSentenceOf = {}

        # Get the set of cleaned sentences from the document
        cleanSet = get_sentences(doc, sentences, clean, originalSentenceOf)

        scores = {}
        
        # For each cleaned sentence, calculate its score and add it to the scores dictionary
        for data in clean:
            temp_doc = cleanSet - set([data])
            score = calculateSimilarity(data, list(temp_doc))
            scores[data] = score

        # Calculate the number of sentences to include in the summary
        n = 20 * len(sentences) / 100

        summarySet = []
        
        while n > 0:
            mmr = {}
            
            # For each sentence, calculate its MMR and add it to the mmr dictionary
            for sentence in scores.keys():
                if not sentence in summarySet:
                    mmr[sentence] = alpha * scores[sentence] - (1-alpha) * calculateSimilarity(sentence, summarySet)	
            
            if mmr == {}:
                break
            
            selected = max(mmr.items(), key=operator.itemgetter(1))[0]	
            summarySet.append(selected)
            
            n -= 1

        # Get the original form of the sentences in the summary set
        original = [originalSentenceOf[sentence].strip() for sentence in summarySet]
        
        # Return the original sentences
        return original
    except Exception as e:
        # If an exception occurs, return an empty list
        return []

path_to_file = './custom_data.csv'
import pandas as pd
df = pd.read_csv(path_to_file)
df['concat_doc'] = df['doc1'] + ' ' + df['doc2'] + ' ' + df['doc3']
df.drop(['doc1', 'doc2', 'doc3'], axis=1, inplace=True)
df

# %%
# Import necessary libraries
import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

for alpha in [0.8]:
    df['mmr'] = ''

    # Write the header to the file
    df.drop(columns=['name', 'concat_doc']).iloc[0:0].to_csv(f'test_{alpha}.csv', index=False)

    for i, row in tqdm(df.iterrows()):
        df.at[i, 'mmr'] = get_mmr(df.at[i, 'concat_doc'], alpha)
        
        # If the MMR is an empty list, skip this row
        if df.at[i, 'mmr'] == []:
            continue

        row = df.iloc[i].drop(['concat_doc', 'name'])

        # Save the current row to the file
        row.to_frame().T.to_csv(f'test_{alpha}.csv', mode='a', header=False, index=False)

pth = "test_0.8.csv"
mmr_source = pd.read_csv(pth)
mmr_s = mmr_source['mmr'][0]
mmr_s = str(eval(mmr_s))
print(mmr_s)
print("\n\n MMR Completed. Generate... \n\n")

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(device)

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
hf_auth = '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'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_auth_token=hf_auth)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)
print(f"Model size: {model.get_memory_footprint():,} bytes")

DEFAULT_SYSTEM_PROMPT = """\
You are an agent generates a summary using key ideas and concepts. You have to ensure that the summaries are coherent, fluent, relevant and consistent.
"""
instruction = """
"""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS  # system prompt: Default instruction to be given to the model
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST  # Final Template: takes in instruction as well.
    # Here it would take in the summary and the source
    return prompt_template


# Function to remove the prompt from the final generated answer
def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text


def remove_substring(string, substring):
    return string.replace(substring, "")


def generate(text):
    prompt = get_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs,
                             max_new_tokens=512,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.eos_token_id,
                             )
    final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    final_outputs = cut_off_text(final_outputs, '</s>')
    final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs  # , outputs


def parse_text(text):
    wrapped_text = textwrap.fill(text, width=100)
    print(wrapped_text + '\n\n')
    # return wrapped_text


# Using pipeling to activate LangChain
from transformers import pipeline

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens=512,
                # do_sample=True,
                # top_k=30,
                # num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                )

llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

system_prompt = """
You are an agent who is tasked with creating cohesive and relevant text that integrates key ideas and concepts provided in a list format. Your goal is to produce summaries that are fluent, coherent, and consistent.
"""

instruction = """
Generate the required text using the list of key ideas and concepts provided below:\n{text}
"""
# Loading and setting the system prompt
template = get_prompt(instruction, system_prompt)
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

generated_output = llm_chain.run(mmr_s)
print("Generated Output: \n\n")
print(generated_output)
