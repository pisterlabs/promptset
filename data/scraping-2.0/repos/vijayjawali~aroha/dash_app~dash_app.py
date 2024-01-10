import math
import dash
from rouge import Rouge
from dash import dcc as dcc
from dash import html as html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output, State
import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from transformers import pipeline

import nltk
import re
import math
import contractions
import string
import pickle
import json
import networkx as nx
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

config_file = args.config


def findField(file, fieldName, allLines = []) -> str:
    """
    
    finds the fields from the given input and returns a line in string format if the key is available in file else returns an empty string

    Args:
        file : input file to be processed
        fieldName : name of key to search in file
        allLines : an array of line string, each element corresponds to a single line in the file. Defaults to [].

    Returns:
        str: a line in string form containing the fieldname that is being searched int he file, empty when field name is not found
    
    >>> Example: 
        file
            ID,VM0010_Viso 
            Name,Subj010_ 
            Age,  0y     
    >>> findField(file, 'ID')
    >>> ,VM0010_Viso
    >>> findField(file, 'address)
    >>> ""
    """
    if not allLines:
        allLines = file.readlines()
            
    
    for currentLine in allLines:
        if fieldName in currentLine:
            currentLine = currentLine.split(fieldName)
            return  ''.join(currentLine[1])
    
    return ""

config = open(config_file, 'r', errors="ignore")

named_entities_path = findField(config, "named_entities").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

bart_tokenizer_cache_dir_path = findField(config, "bart_tokenizer_cache_dir_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

bart_model_cache_dir_path = findField(config, "bart_model_cache_dir_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

t5_tokenizer_cache_dir_path = findField(config, "t5_tokenizer_cache_dir_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

t5_model_cache_dir_path = findField(config, "t5_model_cache_dir_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

t5_fine_tuned_model_path = findField(config, "t5_fine_tuned_model_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

llama2_access_token_path = findField(config, "llama2_access_token_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

llama2_tokenizer_cache_dir_path = findField(config, "llama2_tokenizer_cache_dir_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

llama2_model_cache_dir_path = findField(config, "llama2_model_cache_dir_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

peft_model_path = findField(config, "peft_model_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

seq2seq_encoder_model_path = findField(config, "seq2seq_encoder_model_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

seq2seq_decoder_model_path = findField(config, "seq2seq_decoder_model_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

summary_tokenizer_path = findField(config, "summary_tokenizer_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

article_tokenizer_path = findField(config, "article_tokenizer_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

summary_vocabulary_path = findField(config, "summary_vocabulary_path").lstrip(',').replace('\n','').lstrip().rstrip()
config.seek(0)

# read all parquet files from a folder containing multiple parquet files in pandas dataframe
named_entities = pd.read_parquet(named_entities_path, engine='pyarrow')

# seperate the list of named entities into seperate rows
explode_df = named_entities.explode("named_entities").rename(columns={"named_entities": "entity"})

# get the id and entity columns only
entity_df = explode_df[['id', 'entity']]

# retrieve the entity name from the entity column
entity_df["entity_name"] = entity_df["entity"].str[1]

# create a list of unique entities from entity_name column in entity_df
unique_entities = entity_df['entity_name'].str.lower().unique().tolist()

# a function that takes an input string and returns a list of ids that contain the input string from the entity_df dataframe
def get_ids(input_string):
    # Replace NaN values in 'entity_name' column with empty strings
    entity_df['entity_name'] = entity_df['entity_name'].fillna('')

    # Filter the DataFrame based on the exact input_string and return unique IDs as a list
    return entity_df[entity_df['entity_name'].str.lower() == input_string.lower()]['id'].unique().tolist()

# a function that takes a list of ids and returns a list of articles that contain the ids from the named_entities dataframe
def get_articles(ids):
    # Filter the DataFrame based on the ids list and return unique articles as a list
    return named_entities[named_entities['id'].isin(ids)]['article'].unique().tolist()

# initialize the BART tokenizer and model
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=bart_tokenizer_cache_dir_path)
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", cache_dir=bart_model_cache_dir_path)

t5_tokenizer = T5Tokenizer.from_pretrained("t5-base", cache_dir=t5_tokenizer_cache_dir_path)
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base", cache_dir=t5_model_cache_dir_path)

t5_fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(t5_fine_tuned_model_path)


llama2_model_name = "meta-llama/Llama-2-7b-chat-hf"
llama2_access_token = llama2_access_token_path
    
llama2_tokenizer = AutoTokenizer.from_pretrained(llama2_model_name, 
                                        use_auth_token=llama2_access_token, 
                                        cache_dir=llama2_tokenizer_cache_dir_path
                                        )

llama2_model = AutoModelForCausalLM.from_pretrained(llama2_model_name, 
                                            use_auth_token=llama2_access_token, 
                                            cache_dir=llama2_model_cache_dir_path)


peft_model = PeftModel.from_pretrained(llama2_model, peft_model_path)

seq2seq_encoder_model = load_model(seq2seq_encoder_model_path)
seq2seq_decoder_model = load_model(seq2seq_decoder_model_path)

def get_summary(article):
    """
    Generate a summary of the given article using the BART model.

    Parameters:
        article (str): The input article text.

    Returns:
        str: The generated summary of the article.
    """
    input_text = article
    max_length = math.ceil(len(article)/10)

    if max_length > 1024:
        max_length = 1024
    elif max_length < 56:
        max_length = 56

    inputs = bart_tokenizer(input_text, truncation=True, padding="longest", max_length=1024, return_tensors="pt")
    summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, max_length=max_length, early_stopping=True)

    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# a function that takes a list of articles and returns a list of summaries of the articles
def get_summaries(articles):
    """
    Generate summaries for a list of articles using the BART model.

    Parameters:
        articles (list[str]): A list of articles, where each article is a string.

    Returns:
        list[str]: A list of summaries corresponding to each input article.
    """
    # Create an empty list to store summaries
    summaries = []

    # Loop through all articles and append the summary to the list
    for article in articles:
        summaries.append(get_summary(article))

    return summaries

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_tf_idf_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    preprocessed_text = contractions.fix(preprocessed_text)
    return preprocessed_text

def tf_score(word, sentence):
    word_frequency_in_sentence = sentence.split().count(word)
    len_sentence = len(sentence.split())
    tf = word_frequency_in_sentence / len_sentence
    return tf

def idf_score(no_of_sentences, word, sentences):
    no_of_sentence_containing_word = sum(1 for sentence in sentences if word in sentence)
    idf = math.log10(no_of_sentences / (no_of_sentence_containing_word + 1))
    return idf

def get_tf_idf_summary(custom_text, percentage):

    no_of_sentences = math.ceil(len(sent_tokenize(custom_text)) * (percentage/100))
    sentences = sent_tokenize(custom_text)

    sentences_tf_idf = {}
    for i, sentence in enumerate(sentences, 1):
        sentence_tf_idf = 0
        sentence = re.sub(r'\d+', '', sentence)
        pos_tagged_sentence = nltk.pos_tag(sentence.split())
        for word, pos_tag in pos_tagged_sentence:
            if word.lower() not in stop_words and len(word) > 1 and pos_tag.startswith(('NN', 'VB')):
                word = lemmatizer.lemmatize(word.lower())
                tf = tf_score(word, sentence)
                idf = idf_score(len(sentences), word, sentences)
                tf_idf = tf * idf
                sentence_tf_idf += tf_idf
        sentences_tf_idf[i] = sentence_tf_idf

    sentences_tf_idf = sorted(sentences_tf_idf.items(), key=lambda x: x[1], reverse=True)


    summary = []
    sentence_no = [x[0] for x in sentences_tf_idf[:no_of_sentences]]
    sentence_no.sort()

    for i, sentence in enumerate(sentences, 1):
        if i in sentence_no:
            summary.append(sentence)
    return " ".join(summary)

def preprocess_text_rank_lsa_text(text):
    text = text.lower()
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = []
    for sentence in sentences:
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        sentence = re.sub(r"[^\w\s]", "", sentence)
        sentence = contractions.fix(sentence)
        tokens = word_tokenize(sentence)
        tokens = [token for token in tokens if token not in stop_words]
        preprocessed_sentences.append(tokens)
    return preprocessed_sentences, sentences

def build_similarity_matrix(sentences):
    sentence_vectors = []
    for sentence in sentences:
        sentence_vectors.append(' '.join(sentence))
    vectorizer = TfidfVectorizer().fit_transform(sentence_vectors)
    similarity_matrix = cosine_similarity(vectorizer)
    return similarity_matrix

def get_text_rank_summary(custom_text, percentage):
    processed_article,sentence_tokens = preprocess_text_rank_lsa_text(custom_text)
    similarity_matrix = build_similarity_matrix(processed_article)
    top_n=math.ceil(len(sentence_tokens) * (percentage/100))
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentence_tokens)), reverse=True)
    sentence_array = [sentence[1] for sentence in ranked_sentences[:top_n]]
    return ''.join([''.join(sentence) for sentence in sentence_array])

def get_topic_tf_idf_summary(summaries, topic):
    
    processed_summaries = [preprocess_tf_idf_text(summary) for summary in summaries]
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([topic] + processed_summaries)
    cosine_similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()
    sorted_indices = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)
    ranked_summaries = [(summaries[i], cosine_similarities[i]) for i in sorted_indices]
    return "\n\n".join([f"{summary}" for rank, (summary, similarity) in enumerate(ranked_summaries, start=1)])

def get_lsa_summary(custom_text, percentage):
    processed_article,sentence_tokens = preprocess_text_rank_lsa_text(custom_text)
    sentences = processed_article
    num_sentences=math.ceil(len(sentence_tokens) * (percentage/100))
    sentence_vectors = []
    for sentence in sentences:
        sentence_vectors.append(' '.join(sentence))
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence_vectors)
    lsa_model = TruncatedSVD(n_components=num_sentences)
    lsa_matrix = lsa_model.fit_transform(tfidf_matrix)
    sentence_scores = lsa_matrix.sum(axis=1)
    ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentence_tokens)), reverse=True)
    sentence_array = [sentence[1] for sentence in ranked_sentences[:num_sentences]]
    return ''.join([''.join(sentence) for sentence in sentence_array])

def get_t5_summary(custom_text, percentage):
    input_text = custom_text
    total_length = len(custom_text)
    summary_length = math.ceil(total_length * (percentage / 100))

    if total_length > 512:
        # Divide the input text into equal parts
        num_parts = math.ceil(total_length / 512)
        text_parts = [input_text[i * 512:(i + 1) * 512] for i in range(num_parts)]
        max_input_length = math.ceil(total_length/num_parts)
        max_length = summary_length/num_parts
        if max_length > 512:
            max_length = 512
        elif max_length < 56:
            max_length = 56
        # Generate summaries for each part
        summaries = []
        for part in text_parts:
            input_ids = t5_tokenizer.encode(part, truncation=True, max_length=max_input_length, return_tensors="pt")
            summary_ids = t5_model.generate(input_ids, max_length=max_length)
            summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        # Combine the summaries
        combined_summary = " ".join(summaries)
    else:
        max_length = summary_length
        if max_length > 512:
            max_length = 512
        elif max_length < 56:
            max_length = 56

        input_ids = t5_tokenizer.encode(input_text, truncation=True, max_length=total_length, return_tensors="pt")
        summary_ids = t5_model.generate(input_ids, max_length=max_length)

        combined_summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return combined_summary

def get_t5_fine_tuned_summary(custom_text, percentage):
    input_text = custom_text
    total_length = len(custom_text)
    summary_length = math.ceil(total_length * (percentage / 100))

    if total_length > 512:
        # Divide the input text into equal parts
        num_parts = math.ceil(total_length / 512)
        text_parts = [input_text[i * 512:(i + 1) * 512] for i in range(num_parts)]
        max_input_length = math.ceil(total_length/num_parts)
        max_length = summary_length/num_parts
        if max_length > 512:
            max_length = 512
        elif max_length < 56:
            max_length = 56
        # Generate summaries for each part
        summaries = []
        for part in text_parts:
            input_ids = t5_tokenizer.encode(part, truncation=True, max_length=max_input_length, return_tensors="pt")
            summary_ids = t5_fine_tuned_model.generate(input_ids, max_length=max_length)
            summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        # Combine the summaries
        combined_summary = " ".join(summaries)
    else:
        max_length = summary_length
        if max_length > 512:
            max_length = 512
        elif max_length < 56:
            max_length = 56

        input_ids = t5_tokenizer.encode(input_text, truncation=True, max_length=total_length, return_tensors="pt")
        summary_ids = t5_fine_tuned_model.generate(input_ids, max_length=max_length)

        combined_summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return combined_summary

def get_bart_summary(custom_text, percentage):
    input_text = custom_text
    total_length = len(custom_text)
    summary_length = math.ceil(total_length * (percentage / 100))

    if total_length > 1024:
        # Divide the input text into equal parts
        num_parts = math.ceil(total_length / 1024)
        text_parts = [input_text[i * 1024:(i + 1) * 1024] for i in range(num_parts)]
        max_length = summary_length/num_parts
        if max_length > 1024:
            max_length = 1024
        elif max_length < 56:
            max_length = 56
        # Generate summaries for each part
        summaries = []
        for part in text_parts:
            inputs = bart_tokenizer(part, truncation=True, padding="longest", max_length=1024, return_tensors="pt")
            summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, max_length=max_length, early_stopping=True)
            summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        # Combine the summaries
        combined_summary = " ".join(summaries)
    else:
        max_length = summary_length
        if max_length > 1024:
            max_length = 1024
        elif max_length < 56:
            max_length = 56
        inputs = bart_tokenizer(input_text, truncation=True, padding="longest", max_length=1024, return_tensors="pt")
        summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, max_length=max_length, early_stopping=True)

        combined_summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


    return combined_summary

def get_llama2_summary(custom_text, percentage):

    device = 0 if torch.cuda.is_available() else -1

    pipeline = transformers.pipeline(
        "text-generation",
        model=llama2_model,
        tokenizer=llama2_tokenizer,
        device=device,
        max_length=1024,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=llama2_tokenizer.eos_token_id,
        use_auth_token=llama2_access_token,
        return_text=False
    )

    template = """
            Write a concise summary of the following text delimited by triple backquotes.
            Return your response in paragraph form which covers the key points of the text.
            Add "<start>" at the beginning of the summary and "<end>" at the end of summary.
            Keep the length of summary {percent} percent of the original article
            ```{text}```
        """

    prompt = PromptTemplate(template=template, input_variables=["text", "percent"])
    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    input_args = {'text': custom_text, 'percent': percentage}
    return llm_chain.run(input_args)


def pre_process_llama2_fine_tuned_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^\w\s]", "", text)
    return text

def post_process_llama2_fine_tuned_text(text):
    idx = text.find("### Summary:")
    if idx != -1:
        text = text[idx+len("### Summary:"):]
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    unique_sentences = list(dict.fromkeys(sentences))
    return ' '.join(unique_sentences)

def get_llama2_fine_tuned_summary(custom_text, percentage):

    # Tokenize and truncate the sample_article to 500 tokens
    tokens = llama2_tokenizer.tokenize(pre_process_llama2_fine_tuned_text(custom_text))
    truncated_tokens = tokens[:500]
    truncated_article = '### Article:' + llama2_tokenizer.decode(llama2_tokenizer.convert_tokens_to_ids(truncated_tokens)) + '\n ### Summary:'

    output_length = math.ceil(500 * (percentage / 100))
    
    device = 0
    pipe = pipeline(task="text-generation", model=peft_model.to(device), tokenizer=llama2_tokenizer, device=device, max_length=1000)

    # Generate the summary using the truncated article
    result = pipe(truncated_article)

    # Extract and print the generated output
    generated_output = result[0]['generated_text']
    return llama2_tokenizer.decode(llama2_tokenizer.convert_tokens_to_ids(llama2_tokenizer.tokenize(post_process_llama2_fine_tuned_text(generated_output))[:output_length]))



def preprocess_seq2seq_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\.(?=[^ \W\d])', '. ', str(text))
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([single_word.strip() for single_word in text.split()])
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, wordnet.VERB) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    preprocessed_text = contractions.fix(preprocessed_text)
    return preprocessed_text


def decode_sequence(input_seq, summary_tokenizer, article_tokenizer, summary_vocabulary):
    summary_max_len = 50
    # Encode the input as state vectors
    e_out, e_h, e_c = seq2seq_encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1,1))

    # Populate the first word of target sequence with the start word
    target_seq[0, 0] = summary_tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = seq2seq_decoder_model.predict([target_seq, input_seq, e_out] + [e_h, e_c])

        # Sample a token
        final_prob_index = np.argmax(output_tokens[0, -1, :])

        # Distinguish between generated and copied words based on final_prob_index
        if final_prob_index < len(summary_vocabulary):
            # It's a generated word
            sampled_token = summary_tokenizer.index_word[final_prob_index]
        else:
            # It's a copied word
            copied_word_index = final_prob_index - len(summary_vocabulary)
            sampled_token = article_tokenizer.index_word[input_seq[0, copied_word_index]]

        if(sampled_token!='<end>'):
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word
        if (sampled_token == '<end>' or len(decoded_sentence.split()) >= (summary_max_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = summary_tokenizer.word_index[sampled_token] if sampled_token in summary_tokenizer.word_index else summary_tokenizer.word_index['<unknown>']

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def get_seq2seq_summary(custom_text, percentage):
    with open(summary_tokenizer_path, 'rb') as handle:
        summary_tokenizer = pickle.load(handle)

    with open(article_tokenizer_path, 'rb') as handle:
        article_tokenizer = pickle.load(handle)
    
    with open(summary_vocabulary_path, 'r') as file:
        summary_vocabulary = json.load(file)

    custom_text = preprocess_seq2seq_text(custom_text)
    
    new_word = "<unknown>"
    index = 0
    summary_tokenizer.word_index[new_word] = index
    summary_tokenizer.index_word[index] = new_word
    
    article_max_len = 500
    article = custom_text
    article = '<start> ' + article + ' <end>'
    sequence = article_tokenizer.texts_to_sequences([article])
    sequence_padded = pad_sequences(sequence, maxlen=article_max_len, padding='post')
    return decode_sequence(sequence_padded, summary_tokenizer, article_tokenizer, summary_vocabulary)

# Function to get the article content from the selected suggestion
def get_article_content(suggestion_value, visible_paragraphs):
    if suggestion_value:
        ids = get_ids(suggestion_value)
        article_content = '\n\n'.join(get_summaries(get_articles(ids[:visible_paragraphs])))
        return article_content, ids
    return None, None

def get_updated_article_content(ids, visible_paragraphs):
    if ids:
        article_content = '\n\n'.join(get_summaries(get_articles(ids[visible_paragraphs -3 :visible_paragraphs])))
        return article_content
    return None

# Function to split the article into paragraphs
def split_into_paragraphs(article_content):
    paragraphs = article_content.split('\n\n')
    return paragraphs

def calculate_rouge_scores(original_text, summary_text):
    rouge = Rouge()
    scores = rouge.get_scores(summary_text, original_text, avg=True)
    return scores

def get_topic_summaries(suggestion_value, percentage, num_articles, model):
    if suggestion_value:
        ids = get_ids(suggestion_value)[-num_articles:]
        articles = get_articles(ids)
        summaries = []
        for article in articles:
            if model == 'term frequency inverse document frequency':
                summaries.append(get_tf_idf_summary(article, percentage))
            elif model == 'text rank':
                summaries.append(get_text_rank_summary(article, percentage))
            elif model == 'latent semantic analysis':
                summaries.append(get_lsa_summary(article, percentage))
            elif model == 't5':
                summaries.append(get_t5_summary(article, percentage))
            elif model == 'bart':
                summaries.append(get_bart_summary(article, percentage))
            elif model == 't5 fine tuned':
                summaries.append(get_t5_fine_tuned_summary(article, percentage))
            elif model == 'llama2':
                summaries.append(get_llama2_summary(article, percentage))
            elif model == 'llama2 fine tuned':
                summaries.append(get_llama2_fine_tuned_summary(article, percentage))
            elif model == 'Seq2Seq':
                summaries.append(get_seq2seq_summary(article, percentage))          
        return summaries
    return None

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
]

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

# Number of paragraphs to show initially
initial_paragraphs = 3

app.layout = html.Div([
    html.H1(children='Event based multi-document text summarisation of news articles',
            style={'textAlign': 'center', 'color': '#000205'}),
    html.Div(children=[
        html.Div(children='''Vijay Jawali''', style={'textAlign': 'left', 'color': '#000205'}),
        html.Div(children='''Project - Data Science MSc [06 32255]''', style={'textAlign': 'center', 'color': '#000205'}),
        html.Div(children='''2437649''', style={'textAlign': 'right', 'color': '#000205'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),

    html.Br(),
    html.Div(children='''Supervised by : Dr. Michael Oakes''', style={'textAlign': 'center', 'color': '#000205'}),
    html.Br(),
    html.Hr(style={'border-top': '1px solid #000000', 'width': '100%'}),
    html.Br(),


    html.H3(children='Select an entity to summarise the news events using BART model',
            style={'textAlign': 'center', 'color': '#000205'}),
    html.H4(children='Search Entity : ',
            style={'color': '#000205'}),

    html.Div([
        dcc.Input(
            id='my-input',
            type='text',
            value='',
            placeholder='Type here...',
            autoComplete='off',
            style={'float': 'left', 'width': '20%', 'margin-right': '20px'}
        ),
        html.Br(),
        html.Br(),

        dcc.Loading(
            id="loading-suggestions-container",
            type="circle",
            children=[html.Div(id='suggestions-container', style={'float': 'left', 'width': '20%'})],
            style={'textAlign': 'left', 'width': '20%'}
        ),
    ]),

    html.Div(id='entity-selected', style={'text-align': 'center', 'margin': '0 auto'}),

    html.Br(),

    html.Div([
        dcc.Loading(
            id="loading-article-content",
            type="default",
            children=[
                dcc.Markdown(
                    id='article-content',
                    children='',
                    style={'display': 'none'},
            )]
        ),
        dcc.Markdown(
            id='article-content-updated',
            children='',
        ),
        html.Br(),
        html.Br(),
        dcc.Loading(
            id="loading-article-content-updated",
            type="default",
            children=[
                dcc.Markdown(
                    id='article-content-updated-loading',
                    children='',
                    style={'display': 'none'},
                )]
        ),
        html.Br(),
        html.Button('Load More', id='load-more-button', n_clicks=0, style={'margin': '10px', 'display': 'none'}),

        dcc.Store(id='hidden-paragraphs', data=[]),
        dcc.Store(id='visible-paragraphs', data=initial_paragraphs),
        dcc.Store(id='article-ids', data=None),
    ], style={'float': 'right', 'width': '70%'}),
    html.Br(),
    html.Br(),
        html.Br(),
    html.Hr(style={'border-top': '1px solid #000000', 'width': '100%'}),
    html.Br(),
    html.H3(children='Select an entity to get topic summary',
            style={'textAlign': 'center', 'color': '#000205'}),
    html.H4(children='Search Entity : ',
            style={'color': '#000205'}),
    html.Div(children=[
    html.Div([
        dcc.Input(
            id='my-topic-input',
            type='text',
            value='',
            placeholder='Type here...',
            autoComplete='off',
            style={'float': 'left', 'width': '100%', 'margin-right': '20px'}
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        dcc.Loading(
            id="loading-topic-suggestions-container",
            type="circle",
            children=[html.Div(id='topic-suggestions-container', style={'float': 'left', 'width': '100%'})],
            style={'textAlign': 'left', 'width': '100%'}
        ),
    ]),
    html.Div([
        html.Div(id='topic-entity-selected', style={'text-align': 'center', 'margin': '0 auto'}),
        html.H5(children='Select percentage of text to retain in the summary',
                style={'textAlign': 'center', 'color': '#000205'}),
        dcc.Slider(
            id='topic-percentage-slider',
            min=10, 
            max=90,  
            step=1,  
            value=None,  
            marks={i: f"{i}%" for i in range(10, 91, 10)},  
            included=False,  
            tooltip={'placement': 'bottom'}  
        ),
        html.Br(),
            html.H5(children='Select model',
            style={'textAlign': 'center', 'color': '#000205'}),
        dcc.Dropdown(
            id='topic-model-selection-dropdown',
            options=[
                {'label': 'term frequency inverse document frequency', 'value': 'term frequency inverse document frequency'},
                {'label': 'text rank', 'value': 'text rank'},
                {'label': 'latent semantic analysis', 'value': 'latent semantic analysis'},
                {'label': 't5', 'value': 't5'},
                {'label': 't5 fine tuned', 'value': 't5 fine tuned'},
                {'label': 'bart', 'value': 'bart'},
                {'label': 'llama2', 'value': 'llama2'},
                {'label': 'llama2 fine tuned', 'value': 'llama2 fine tuned'},
                {'label': 'Seq2Seq', 'value': 'Seq2Seq'},
            ],
            value='',
            style={'width': '50%', 'margin': '0 auto'},
        ),
        html.Br(),
        html.H5(children='Select size of the summary',
                style={'textAlign': 'center', 'color': '#000205'}),
        dcc.Dropdown(
        id='topic-length-selection-dropdown',
        options=[
            {'label': 'short (5 articles)', 'value': 'short'},
            {'label': 'medium (10 articles)', 'value': 'short'},
            {'label': 'long (15 articles)', 'value': 'long'},
        ],
        value='',
        style={'width': '50%', 'margin': '0 auto'},
        ),
        html.Br(),
        dcc.Input(id="email-input", type="email", placeholder="Enter your email"),
        html.Div(id="validation-output"),
        html.Br(),
        html.Button("Submit", id="submit-button", disabled=True, n_clicks=0),
        html.Br(),
        html.Div(id='topic-email-confirmation', style={'text-align': 'center', 'margin': '0 auto'}),
        html.Br(),
        dcc.Loading(
            id="loading-topic-email-summary",
            type="default",
            children=[html.Div(id='topic-email-summary',children='', style={'text-align': 'center', 'margin': '0 auto'})],
            style={'textAlign': 'left', 'width': '70%'}
        ),
        ], style={'float': 'right', 'width': '70%', 'margin-left': '20px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    dcc.Store(id='selected-topic-value', data=None),
    dcc.Store(id='generate-topic-summary', data=False),
    html.Br(),
    html.Br(),
    html.Hr(style={'border-top': '1px solid #000000', 'width': '100%', 'margin-top': '20px', 'margin-bottom': '20px'}),
    html.Br(),
    html.Br(),

    html.H3(children='Summarise your own text document using your own model choice',
            style={'textAlign': 'center', 'color': '#000205'}),
    html.H5(children='Select model',
            style={'textAlign': 'center', 'color': '#000205'}),
    dcc.Dropdown(
        id='model-selection-dropdown',
        options=[
            {'label': 'term frequency inverse document frequency', 'value': 'term frequency inverse document frequency'},
            {'label': 'text rank', 'value': 'text rank'},
            {'label': 'latent semantic analysis', 'value': 'latent semantic analysis'},
            {'label': 't5', 'value': 't5'},
            {'label': 't5 fine tuned', 'value': 't5 fine tuned'},
            {'label': 'bart', 'value': 'bart'},
            {'label': 'llama2', 'value': 'llama2'},
            {'label': 'llama2 fine tuned', 'value': 'llama2 fine tuned'},
            {'label': 'Seq2Seq', 'value': 'Seq2Seq'},
        ],
        value='',
        style={'width': '50%', 'margin': '0 auto'},
    ),
    html.H5(children='Select percentage of text to retain in the summary',
            style={'textAlign': 'center', 'color': '#000205'}),
    dcc.Slider(
        id='summary-percentage-slider',
        min=10,
        max=90,
        step=1,
        value=None,
        marks={i: f"{i}%" for i in range(10, 91, 10)},
        included=False,
        tooltip={'placement': 'bottom'}
    ),
    html.H4(id='custom-summary', style={'textAlign': 'center', 'color': '#000205', 'display': 'none'}),


    html.H4(children='Enter your text : ',
            style={'color': '#000205'}),

    dcc.Textarea(
        id='custom-text-input',
        placeholder='Type here...',
        style={'width': '100%', 'height': '300px'}
    ),
    html.Div(id='custom-text-length-info', style={'margin-top': '10px'}),

    html.Br(),
    html.Button('Get Summary', id='get-summary-button', n_clicks=0, style={'margin': '10px'}),
    html.Br(),
    html.H4(children='Summary : ',
            style={'color': '#000205'}),
    html.Br(),
    dcc.Loading(
        id="loading-summary-output",
        type="circle",
        children=[html.Div(id="summary-output", style={'whiteSpace': 'pre-line'})]
    ),
    html.Br(),
    html.Hr(style={'border-top': '1px solid #000000', 'width': '100%', 'margin-top': '20px', 'margin-bottom': '20px'}),
    html.Br(),

    html.Div(
    html.Button('Show Analytics', id='show-analytics-button', n_clicks=0),
    style={'display': 'flex', 'justify-content': 'center', 'margin': '20px'}
    ),

    dcc.Loading(
        id='loading-rouge-scores',
        type='default',
        children=[
            html.Table(id='rouge-scores-table', style={'margin': 'auto'}),
        ]
    ),
    html.Br(),
    html.Hr(style={'border-top': '1px solid #000000', 'width': '100%', 'margin-top': '20px', 'margin-bottom': '20px'}),
    html.Br(),
    html.Br(),
])


@app.callback(
    Output('suggestions-container', 'children'),
    [Input('my-input', 'value')],
    [State('my-input', 'id')]
)
def update_suggestions(value, input_id):
    if value:
        filtered_suggestions = [s for s in unique_entities if str(s).lower().startswith(value.lower())]
        if filtered_suggestions:
            # Check if the number of filtered suggestions is more than 10
            if len(filtered_suggestions) > 10:
                # Wrap the RadioItems in a Div with scrollable style
                return html.Div(
                    dcc.RadioItems(
                        id={'type': 'suggestion', 'index': 'ALL'},
                        options=[{'label': str(s), 'value': str(s)} for s in filtered_suggestions],
                        labelStyle={'display': 'block', 'margin-bottom': '5px'},
                        value=''
                    ),
                    style={'max-height': '500px', 'overflow': 'scroll'} 
                )
            else:
                # Display all suggestions if they are less than or equal to 10
                return dcc.RadioItems(
                    id={'type': 'suggestion', 'index': 'ALL'},
                    options=[{'label': str(s), 'value': str(s)} for s in filtered_suggestions],
                    labelStyle={'display': 'block', 'margin-bottom': '5px'},
                    value=''
                )
    return None


@app.callback(
    Output('entity-selected', 'children'),
    [Input({'type': 'suggestion', 'index': 'ALL'}, 'value')]
)
def update_output(suggestion_value):
    if suggestion_value:
        return html.H3('Entity Selected is: ' + str(suggestion_value),
                       style={'textAlign': 'center', 'color': '#000205'})
    return None

@app.callback(
    Output('article-content', 'children'),
    Output('article-ids', 'data'),
    Output('load-more-button', 'style'),
    [Input({'type': 'suggestion', 'index': 'ALL'}, 'value')],
    [State('visible-paragraphs', 'data')]
)
def update_article_content(suggestion_value, visible_paragraphs):
    if suggestion_value:
        article_content, ids = get_article_content(suggestion_value, visible_paragraphs)
        if article_content:
            paragraphs = split_into_paragraphs(article_content)
            return '\n\n'.join(paragraphs), ids, {'display': 'block'}
    return None, None, {'display': 'none'}

@app.callback(
    [Output('hidden-paragraphs', 'data'),
     Output('visible-paragraphs', 'data'),
     Output('article-content-updated', 'children'), Output('article-content-updated-loading', 'children')],
    [Input('load-more-button', 'n_clicks'),
    Input('article-ids', 'data')],
    [State('hidden-paragraphs', 'data'),
     State('visible-paragraphs', 'data'),
     State('article-content', 'children')]
)
def load_more_content(n_clicks, ids, hidden_paragraphs, visible_paragraphs, article_content):
    if article_content is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if n_clicks is None:
        return hidden_paragraphs, visible_paragraphs, None, None
    num_visible_paragraphs =  initial_paragraphs
    if n_clicks > 0:
        num_visible_paragraphs = visible_paragraphs + initial_paragraphs
        updated_article_content = get_updated_article_content(ids, num_visible_paragraphs)
        if updated_article_content:
            article_content =  updated_article_content
    paragraphs = split_into_paragraphs(article_content)
    hidden_paragraphs_to_show = hidden_paragraphs + paragraphs[-3:]
    return hidden_paragraphs_to_show, num_visible_paragraphs, '\n\n'.join(hidden_paragraphs_to_show), None




@app.callback(
    Output('custom-summary', 'children'),
    Output('custom-summary', 'style'),
    [Input('model-selection-dropdown', 'value'),
     Input('summary-percentage-slider', 'value')],
)
def show_custom_selection(model, percentage):
    if (percentage is None) or (model is None):
        return dash.no_update, dash.no_update
    return 'Selected model :' + str(model) + ' , ' + 'Selected percentage :' + str(percentage), {'display': 'block', 'textAlign': 'center', 'color': '#000205'}

@app.callback(
    Output('custom-text-length-info', 'children'),
    [Input('custom-text-input', 'value')]
)
def update_text_length_info(text):
    max_length = 5000
    length = len(text) if text else 0
    warning_style = {'color': 'red'} if length > max_length else {}
    warning_msg = f' ({length}/{max_length})' if length > max_length else f' ({length}/{max_length})'
    return html.Div([
        html.Span('Text Length:', style={'font-weight': 'bold'}),
        html.Span(warning_msg, style=warning_style)
    ])


@app.callback(
    Output('summary-output', 'children'),
    [Input('get-summary-button', 'n_clicks')],
    [State('model-selection-dropdown', 'value'),
     State('summary-percentage-slider', 'value'),
     State('custom-text-input', 'value')]
)
def update_summary(n_clicks, model, percentage, custom_text):
    if custom_text:
        if len(custom_text) > 5000:
            warning_style = {'color': 'red'}
            warning_msg = "Please provide text with less than 5000 characters."
            return html.Div([
                html.Span(warning_msg, style=warning_style)
            ])

    if n_clicks > 0:
        if custom_text and model and percentage:
            if model == 'term frequency inverse document frequency':
                return get_tf_idf_summary(custom_text, percentage)
            elif model == 'text rank':
                return get_text_rank_summary(custom_text, percentage)
            elif model == 'latent semantic analysis':
                return get_lsa_summary(custom_text, percentage)
            elif model == 't5':
                return get_t5_summary(custom_text, percentage)
            elif model == 'bart':
                return get_bart_summary(custom_text, percentage)
            elif model == 't5 fine tuned':
                return get_t5_fine_tuned_summary(custom_text, percentage)
            elif model == 'llama2':
                return get_llama2_summary(custom_text, percentage)
            elif model == 'llama2 fine tuned':
                return get_llama2_fine_tuned_summary(custom_text, percentage)
            elif model == 'Seq2Seq':
                return get_seq2seq_summary(custom_text, percentage)
        else:
            warning_style = {'color': 'red'}
            warning_msg = "Please provide all necessary inputs to generate the summary."
            return html.Div([
                html.Span(warning_msg, style=warning_style)
            ])
    return ''

@app.callback(
    Output('rouge-scores-table', 'children'),
    [Input('show-analytics-button', 'n_clicks')],
    [State('custom-text-input', 'value'),
     State('summary-output', 'children')]
)
def show_rouge_scores(n_clicks, original_text, summary_text):
    if n_clicks > 0 and original_text and summary_text:
        rouge_scores = calculate_rouge_scores(original_text, summary_text)
        headers = ['rouge-1', 'rouge-2', 'rouge-l']
        rows = [html.Tr([
            html.Td(header, style={'font-weight': 'bold', 'border': '2px solid black', 'padding': '10px'}),
            html.Td(round(rouge_scores[header]['f'], 2), style={'border': '1px solid black', 'padding': '10px'}),
            html.Td(round(rouge_scores[header]['p'], 2), style={'border': '1px solid black', 'padding': '10px'}),
            html.Td(round(rouge_scores[header]['r'], 2), style={'border': '1px solid black', 'padding': '10px'})
        ]) for header in headers]

        table_style = {
            'border-collapse': 'collapse',
            'width': '100%',
            'margin': '10px',
            'border': '1px solid black'
        }

        header_style = {
            'font-weight': 'bold',
            'border': '2px solid black',
            'padding': '10px'
        }

        table = html.Table([
            html.Thead(html.Tr([
                html.Th('Metric', style=header_style),
                html.Th('F1-score', style=header_style),
                html.Th('Precision', style=header_style),
                html.Th('Recall', style=header_style)
            ]))] + rows, style=table_style)

        return table
    return None

@app.callback(
    Output('topic-suggestions-container', 'children'),
    [Input('my-topic-input', 'value')],
    [State('my-topic-input', 'id')]
)
def update_topic_suggestions(value, input_id):
    if value:
        filtered_suggestions = [s for s in unique_entities if str(s).lower().startswith(value.lower())]
        if filtered_suggestions:
            # Check if the number of filtered suggestions is more than 10
            if len(filtered_suggestions) > 10:
                # Wrap the RadioItems in a Div with scrollable style
                return html.Div(
                    dcc.RadioItems(
                        id={'type': 'topic-suggestion', 'index': 'ALL'},
                        options=[{'label': str(s), 'value': str(s)} for s in filtered_suggestions],
                        labelStyle={'display': 'block', 'margin-bottom': '5px'},
                        value=''
                    ),
                    style={'max-height': '500px', 'overflow': 'scroll'}
                )
            else:
                # Display all suggestions if they are less than or equal to 10
                return dcc.RadioItems(
                    id={'type': 'topic-suggestion', 'index': 'ALL'},
                    options=[{'label': str(s), 'value': str(s)} for s in filtered_suggestions],
                    labelStyle={'display': 'block', 'margin-bottom': '5px'},
                    value=''
                )
    return None

@app.callback(
    Output('topic-entity-selected', 'children'),
    Output('selected-topic-value', 'data'),
    [Input({'type': 'topic-suggestion', 'index': 'ALL'}, 'value')]
)
def update_topic_output(suggestion_value):
    if suggestion_value:
        return  html.H3('Entity Selected is: ' + str(suggestion_value),
                       style={'textAlign': 'center', 'color': '#000205'}), suggestion_value
    return None, None
    
@app.callback(
    [Output("validation-output", "children"),
     Output("submit-button", "disabled"),
     Output("submit-button", "style")],
    [Input("email-input", "value")]
)
def validate_email(email):
    if email:
        if re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return html.Div("Valid email", style={"color": "green"}), False, {'border-color': 'green'}
        else:
            return html.Div("Invalid email", style={"color": "red"}), True, {'border-color': 'red'}
    else:
        return "", True, None

@app.callback(
    Output("topic-email-confirmation", "children"),
    Output("submit-button", "n_clicks"),
    Output('generate-topic-summary', 'data'),
    [Input("submit-button", "disabled"),
     Input("email-input", "value"),
     Input('submit-button', 'n_clicks'), 
     Input('selected-topic-value', 'data'), 
     Input('topic-percentage-slider', 'value')]
)
def send_confirmation(email_validation, email, n_clicks, topic, percentage):
    if not email_validation and email and n_clicks > 0 and percentage:
        return html.H5("Email on topic: " + topic + " will be sent to: " + email, style={'textAlign': 'center', 'color': 'green'}), 0, True
    return None, n_clicks, False

@app.callback(
    Output("topic-email-summary", "children"),
    [Input("submit-button", "disabled"),
     Input("email-input", "value"), 
     Input('selected-topic-value', 'data'), 
     Input('generate-topic-summary', 'data'), 
     Input('topic-percentage-slider', 'value'), 
     Input('topic-length-selection-dropdown', 'value'), 
     Input('topic-model-selection-dropdown', 'value')]
)
def get_topic_summary_from_confirmation(email_validation, to_email, topic, generate_topic_summary, percentage, length, model):
    if not email_validation and to_email and topic and generate_topic_summary and percentage and length and model:
        from_email = 'textsummarisationsmtp@gmail.com'
        app_password = 'puuq lbcw nlmc zlcm'
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = 'Topic Summary on: ' + topic + 'with model: ' + model
        if length == 'short':
            number_of_summaries = 5
        elif length == 'medium':
            number_of_summaries = 10
        elif length == 'long':
            number_of_summaries = 15
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(from_email, app_password)
                topic_summaries = get_topic_summaries(topic, percentage, number_of_summaries, model)
                topic_summaries = get_topic_tf_idf_summary(topic_summaries, topic)
                msg.attach(MIMEText(topic_summaries, 'plain'))
                server.sendmail(from_email, to_email, msg.as_string())
                server.quit()
            return html.H5("Email on topic: " + topic + ", sent to: " + to_email + " successfully", style={'textAlign': 'center', 'color': 'green'})
        except Exception as e:
            return html.H5("Failed to send email on topic: " + topic + " to: " + to_email + '\n' + 'Exception: ' + str(e), style={'textAlign': 'center', 'color': 'red'})

if __name__ == '__main__':
    app.run_server()
    # app.run_server(host='localhost', port=8081, debug=True, suppress_callback_exceptions=True)