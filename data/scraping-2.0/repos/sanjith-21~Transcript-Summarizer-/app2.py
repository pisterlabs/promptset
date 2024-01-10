import heapq
from flask import Flask, make_response, redirect, render_template, request, send_from_directory
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi as yta
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai
import pdfkit
import urllib.parse
from nltk.cluster.util import cosine_distance

import networkx as nx
openai.api_key = "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"
import os
def summarize(text):
  stopWords = set(stopwords.words("english"))
  words = word_tokenize(text)

  freqTable = dict()
  for word in words:
    word = word.lower()
    if word in stopWords:
      continue
    if word in freqTable:
      freqTable[word] += 1
    else:
      freqTable[word] = 1

  sentences = sent_tokenize(text)
  sentenceValue = dict()

  for sentence in sentences:
    for word, freq in freqTable.items():
      if word in sentence.lower():
        if sentence in sentenceValue:
          sentenceValue[sentence] += freq
        else:
          sentenceValue[sentence] = freq


  sumValues = 0
  for sentence in sentenceValue:
    sumValues += sentenceValue[sentence]

  average = int(sumValues / len(sentenceValue))

  summary = ''
  for sentence in sentences:
    if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.5 * average)):
      summary += " " + sentence
    
  return summary
def generate_summary_final(transcript):
    # Split the transcript into smaller chunks
    max_chunk_size = 4096
    chunks = [transcript[i:i+max_chunk_size] for i in range(0, len(transcript), max_chunk_size)]

    # Generate summaries for each chunk
    summaries = []
    for chunk in chunks:
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Summarize this briefly:\n\n{chunk}",
                temperature=0.7,
                max_tokens=250,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            summary = response.choices[0].text.strip()
            summaries.append(summary)
        except Exception as e:
            print("Error generating summary:", str(e))
            continue

    # Join the summaries into a final summary with a total word count limit of 300
    final_summary = " ".join(summaries)
    sentences = re.findall(r"[^.!?]+[.!?]", final_summary)
    sentence_scores = {}
    for sentence in sentences:
        sentence_scores[sentence] = sentence_score(sentence, transcript)
    top_sentences = heapq.nlargest(10, sentence_scores, key=sentence_scores.get)
    final_summary = " ".join(top_sentences)
    
    return final_summary

def sentence_score(sentence, transcript):
    words = sentence.split()
    word_count = len(words)
    match_count = 0
    for word in words:
        if word.lower() in transcript.lower():
            match_count += 1
    return match_count / word_count

def generate_summary(transcript):
     # Split the transcript into sentences using regular expressions
    sentences = re.findall(r"[^.!?]+[.!?]", transcript)
    
    # Initialize variables
    max_chunk_size = 4096
    chunks = []
    current_chunk = ""
    
    # Add sentences to the current chunk until it reaches the maximum size
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Generate summaries for each chunk
    summaries = []
    for chunk in chunks:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Summarize this in 300 words:\n\n{chunk}",
            temperature=0.7,
            max_tokens=200,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        summary = response.choices[0].text.strip()
        summaries.append(summary)
        
    # Join the summaries into a final summary
    final_summary3 = " ".join(summaries)
    
    return final_summary3
def read_article(text):
    article = text.split(". ")
    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return sentences

# Function to calculate sentence similarity using cosine similarity
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)
# Function to create similarity matrix
def build_similarity_matrix(sentences, stopwords=None):
    if stopwords is None:
        stopwords = []

    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue

            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords)

    return similarity_matrix
# Function to generate summary
def generate_summary2(text, num_sentences=50, cutoff=0.1):
    sentences = nltk.sent_tokenize(text)
    word_frequencies = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies.keys():
                if len(sentence.split(' ')) < 30:
                    if i not in sentence_scores:
                        sentence_scores[i] = word_frequencies[word]
                    else:
                        sentence_scores[i] += word_frequencies[word]

    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary_sentences.sort()
    
    while len(summary_sentences) < 10:
        num_sentences += 1
        summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        summary_sentences.sort()

    summary = ' '.join([sentences[i] for i in summary_sentences])

    return summary

def get_transcript(video_link):
    video_id = re.search(r'v=([^&]*)', video_link).group(1)
    transcript = yta.get_transcript(video_id)
    lines = ""
    temp = ""
    for line in transcript:
      temp = line['text']
      lines += temp.replace("\n", " ")
    with open("clean.txt", "w") as f:
      f.write(lines)

app = Flask(__name__)
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    options = {
         'page-size': 'Letter',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'custom-header': [
            ('Accept-Encoding', 'gzip')
        ],
        'no-outline': None,
        'font-family': 'Calibiri'  # set the font family to Arial
    }
    summary = request.form['summary']
    pdf = pdfkit.from_string(summary, False)
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=summary.pdf'
    return response
    


# whatsapp code

@app.route('/share_whatsapp', methods=['POST'])
def share_whatsapp():
    summary = request.form['summary']
    urlencoded_summary = urllib.parse.quote(summary)
    url = f"whatsapp://send?text={urlencoded_summary}"
    return redirect(url)


@app.route('/', methods=['GET', 'POST'])
def index():
  link = ''
  summary = ""
  final = ""
  if request.method == 'POST':
    link = request.form['link']
    method = request.form['method']
    get_transcript(link)
    with open("clean.txt", "r") as f:
      input_text = f.read()
    now = len(input_text.split())
    if method == 'nlp':
      # summary = summarize(input_text)
      summary = summarize(input_text)
      final=summary
    elif method == 'ml':
      summary =generate_summary_final(input_text)
      if len(summary)>500:
        summary3=generate_summary(summary)
      else:
        summary3=summary
      final = summary3
  return render_template('index.html', value = final)

if __name__ == '__main__':
  app.run("0.0.0.0")