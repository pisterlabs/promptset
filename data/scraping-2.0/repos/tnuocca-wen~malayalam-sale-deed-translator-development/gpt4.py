from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

import ast

import re
import tiktoken

import csv
# Creating Examples list from the examples text file
examples_list = None
def make_examples_list():
  global examples_list
  with open('examples.txt', 'r', encoding='utf8') as tpairs:
    examples_list = tpairs.read()

  examples_list = ast.literal_eval(examples_list)
  print(type(examples_list))
  # examples = [n.strip() for n in examples]

# Calculating the token length of the prompts
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Tokenizing the examples and the input text for similarity checking
def tokenize(text):
    pattern = r'\b[\u0D00-\u0D7F]+'  # Unicode range for Malayalam characters
    return re.findall(pattern, text)
    # encoding = tiktoken.encoding_for_model("gpt-4")
    # return encoding.encode(text)

# similar Examples extraction from the examples list
def calculate_similarity(sentence1, sentence2):
    # print(sentence1)
    tokens1 = set(tokenize(sentence1))
    tokens2 = set(tokenize(sentence2))
    # print("token 1", tokens1, end='\n')
    # print("token 2", tokens2, end='\n')
    common_tokens = tokens1.intersection(tokens2)
    return len(common_tokens)


def find_most_similar(sentences, input_sentence, num_results=3):
    similarities = []
    for entry in sentences:
        malayalam_sent = entry['malayalam_sent']
        translation = entry['translation']
        similarity = calculate_similarity(input_sentence, malayalam_sent) 
        if similarity > 0:  # Ignore entries with similarity 0
            similarities.append({'malayalam_sent': malayalam_sent, 'translation': translation})
        # if len(similarity) == len(tokens2) or len(similarity) > len(tokens2)/2:

    similarities.sort(key=lambda x: calculate_similarity(input_sentence, x['malayalam_sent']), reverse=True)
    
    return similarities[:num_results]


def example_select(sentence):
  # Example input list of dictionaries
  input_list = examples_list
  input_sentence = sentence
  
  # Find the three most similar dictionaries
  most_similar = find_most_similar(input_list, input_sentence, num_results=3)#int(len(examples_list)/2)

  # Print the results
  # for idx, entry in enumerate(most_similar, start=1):
    # print(f"Match {idx}:\nMalayalam Sentence: {entry['malayalam_sent']}\nTranslation: {entry['translation']}\n")
  
  return most_similar


# Phrases extraction from examples list
def calculate_similarity_for_phrases(sentence1, sentence2):
    # print(sentence1)
    tokens1 = set(tokenize(sentence1))
    tokens2 = set(tokenize(sentence2))
    # print("token 1", tokens1, end='\n')
    # print("token 2", tokens2, end='\n')
    common_tokens = tokens1.intersection(tokens2)
    return len(common_tokens)/len(tokens2)

def find_most_similar_phrases(sentences, input_sentence):
    similarities = []
    for entry in sentences:
        malayalam_sent = entry['malayalam_sent']
        translation = entry['translation']
        similarity = calculate_similarity_for_phrases(input_sentence, malayalam_sent)
        if similarity > 0:  # Ignore entries with similarity 0
            similarities.append({'malayalam_sent': malayalam_sent, 'translation': translation})
        # if len(similarity) == len(tokens2) or len(similarity) > len(tokens2)/2:

    similarities.sort(key=lambda x: calculate_similarity_for_phrases(input_sentence, x['malayalam_sent']), reverse=True)

    length = 0
    num_results=0
    for entry in similarities:
      malayalam_sent = entry['malayalam_sent']
      translation = entry['translation']
      length += num_tokens_from_string(malayalam_sent, "cl100k_base")
      length += num_tokens_from_string(translation, "cl100k_base")
      num_results = num_results + 1
      if length > 100:
        break

    print("Total number of similar examples:", len(similarities), "\n", "Total number of taken examples", num_results,"\n", "Total number of tokens", length)
    return similarities[:num_results]

def example_select_phrases(sentence):
  # Example input list of dictionaries
  input_list = examples_list
  input_sentence = sentence

  # Find the three most similar dictionaries
  most_similar = find_most_similar_phrases(input_list, input_sentence)#int(len(examples_list)/2)

  # Print the results
  # for idx, entry in enumerate(most_similar, start=1):
    # print(f"Match {idx}:\nMalayalam Sentence: {entry['malayalam_sent']}\nTranslation: {entry['translation']}\n")

  return most_similar



# Translation function using GPT-4
def chatGPT_Trans(text):
  global examples_list
  make_examples_list()

  example_prompt = PromptTemplate(input_variables=["malayalam_sent", "translation"], template="Translate to English : {malayalam_sent}\n{translation}")

  llm = ChatOpenAI(model = 'gpt-4', temperature=0)
  phrases = ""
  ph = example_select_phrases(text)
  for item in ph:
    phrases += "Translate to English: " + item['malayalam_sent'] + "\n"
    phrases += item['translation'] + "\n\n"

  prompt = FewShotPromptTemplate(
    examples = example_select(text),
    example_prompt=example_prompt,
    suffix='''You are an expert translator of legal documents and sale deeds from malayalam to english
     note to translate
     {phrases}
     in the following: {input}''',
    input_variables=["input","phrases"])
  
  comp_prompt = prompt.format(input=text,phrases=phrases)
  # print(type(comp_prompt))
  print(comp_prompt)
  print(num_tokens_from_string(comp_prompt, "cl100k_base"))
  
  chain = LLMChain(llm=llm, prompt=prompt)
  response = chain.run({"input":text, "phrases":phrases})
  return response


# Translation function using custom fine-tuned model
def chatGPT_Trans_ft(text):
  llm = ChatOpenAI(model = 'ft:gpt-3.5-turbo-0613:personal::86bad9JC', temperature=0)
  # template = '''Return all the named Entities and their english transliteration as python lists. Return two lists one for named entities and the other for
  # english transliteration.

  # Context:" {text}"
  # Response:
  # '''
  # prompt_template = PromptTemplate(
  #     input_variables=["text"],
  #     template=template,
  # )
  make_examples_list()

  phrases = ""
  ph = example_select_phrases(text)
  for item in ph:
    phrases += "Translate to English: " + item['malayalam_sent'] + "\n"
    phrases += item['translation'] + "\n\n"


  example_prompt = PromptTemplate(input_variables=["malayalam_sent", "translation"], template="Translate to English : {malayalam_sent}\n{translation}")

  # dictionary = dictionary_search(text)

  prompt = FewShotPromptTemplate(
    examples = example_select(text),
    example_prompt=example_prompt,
    suffix='''{phrases}
    You are an expert translator of legal documents and sale deeds from malayalam to english.
    Rely entirely on the examples given wherever possible.
     translate: {input}''',
    input_variables=["input","phrases"])

  # prompt = PromptTemplate(input_variables=["input", "phrases"], template='''{phrases}
  # Do not Hallucinate or make up information. Do not append descriptions.
  # Now Translate the entire text from Malayalam to English : {input}''')

  comp_prompt = prompt.format(input=text, phrases=phrases)
  print(comp_prompt)
  print(num_tokens_from_string(comp_prompt, "cl100k_base"))

  chain = LLMChain(llm=llm, prompt=prompt)
  response = chain.run({"input":text, "phrases":phrases})
  return response


