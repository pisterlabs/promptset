"""
This Python script is designed for testing and evaluating the intent-less chatbot's retrieval component.

Key Components and Steps:
- Retriever Configuration: The script sets up two types of retrievers - one using BERT embeddings and the other using
    OpenAI embeddings (Step 2).

- Retrievers Testing: Tests both retrievers on a set of web support queries (Step 3).

- Retrieval Evaluation: Assesses the effectiveness of each retriever in accurately fetching relevant documents (Step 5)
    and calculating precision, recall and F1 score per retriever and fetched number of documents (step 7)
    Precision = Number of relevant retrievals / total number of retrievals
    Recall = Number of relevant retrievals /  total number of relevant documents
    F1 score = 2×((Precision+Recall)/(Precision×Recall)=)

- Visualization: Results are visualized with bar charts, showcasing the rate of correctly answered questions by document
    count for each retriever (Step 6 and 8).

To run this script update the 'path' variable to the root directory of this project and add your OpenAI API key to
'openaiapikey.txt' in the root directory of this project.

For visualization, execute Steps 5 and 8 using the saved result CSV files.
"""


# 1. Set Up-------------------------------------------------------------------------------------------------------------
# Set path to root directory and OpenAI API key
import sys
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis" # Change
testing_path = path + r"\testing"
sys.path.append(testing_path)

from testing_chatbot.testing_functions import open_file
import openai
import os
os.environ["OPENAI_API_KEY"] = open_file(path + "\openaiapikey.txt")
openai.api_key = os.getenv("OPENAI_API_KEY") # Add OpenAI API key to this .txt file

# Standard Libraries
import pandas as pd
import json

# Libraries for running the testing chatbot / for text generation
import importlib
from sentence_transformers import SentenceTransformer
from testing_chatbot.testing_functions import replace_links_with_placeholder, num_tokens_from_string, adjust_similarity_scores

# Libraries for transforming csv data
import ast

# Libraries for visualization
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


# 2. Build the retrievers-----------------------------------------------------------------------------------------------
"""The retriever encodes the input question into a numerical vector using a predefined encoder.
It then calculates the similarity score between the encoded question vector and each document
vector in the loaded index using the dot product.
The retriever returns the top 'n' documents with the highest similarity scores"""

# Load a list of words from a file. These words are checked against questions; if present,
# they trigger a higher ranking for associated documents during retrieval.
words_to_check = ast.literal_eval(open_file(testing_path + r"\testing_data\words_to_check_for_adjusted_similarity_score.txt"))

# Function calculating the dot product.
def similarity(v1, v2):  # returns dot product of two vectors
    return np.dot(v1, v2)


# Defining the retriever that uses BERT embeddings, specifically the embeddings from the model
# 'sentence-transformers/facebook-dpr-question_encoder-single-nq-base' which is a DPR model
def BERT_retriever(query=None, count=20, index_path=None):
    question_model = SentenceTransformer('sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
    vector = question_model.encode(query)
    with open(index_path, 'r') as infile:
        data = json.load(infile)
    scores = []
    for i in data:
        score = similarity(vector, i['vector'])
        scores.append({'context': i['context'], 'score': score, "metadata": i["metadata"]})
    # Comment for unadjusted scores: If certain words or word combinations in the question indicate a specific intent,
    # the scores of documents associated with that intent are multiplied.
    # This increases the likelihood of selecting documents relevant to the question.
    scores = adjust_similarity_scores(results=scores, question=query, word_intent_dict=words_to_check, multiplier=1.2)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    return ordered[0:count]


# This function is used to get the embeddings form the OpenAI model 'text-embedding-ada-002'
def gpt3_embedding(content, engine="text-embedding-ada-002"):
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


# Defining the retriever using OpenAI embeddings, specifically the embeddings from the OpenAI model
# 'text-embedding-ada-002'
def OpenAI_retriever(query=None, count=20, index_path=None):
    vector = gpt3_embedding(query)
    with open(index_path, 'r') as infile:
        data = json.load(infile)
    scores = []
    for i in data:
        score = similarity(vector, i['vector'])
        scores.append({'context': i['context'], 'score': score, "metadata": i["metadata"]})
    # Comment for unadjusted scores: If certain words or word combinations in the question indicate a specific intent,
    # the scores of documents associated with that intent are multiplied. This increases
    # the likelihood of selecting documents relevant to the question.
    scores = adjust_similarity_scores(results=scores, question=query, word_intent_dict=words_to_check, multiplier=1.2)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    return ordered[0:count]


# 3. Use retrievers to retrieve a set of 'n' documents for each question in the testing dataset-------------------------
# Import testing dataset to test the two retrievers on
testing_data_set = pd.read_csv(testing_path + r"\testing_data\test_dataset.csv", encoding="utf-8")
questions = testing_data_set['Beschreibung'].tolist()
# In the index links were replaced with placeholders. To maintain consistency, the same is done to these questions.
questions = [replace_links_with_placeholder(i) for i in questions]
human_responses = testing_data_set['Lösungsbeschreibung'].tolist()
question_ids = testing_data_set['Incident-ID'].tolist()
intents = testing_data_set['intent'].tolist()

# Import OpenAI testing chatbot to test the retriever component.
"""This testing chatbot is specifically designed for evaluating individual chatbot components.
It accepts various parameters, including the test type (retriever, prompt_max_input, memory), the selected retriever,
the chosen prompt, and the number of document tokens to add to the prompt, to systematically assess the chatbot's
performance across these different variables."""

sys.path.append(testing_path + r'\testing_chatbot')
module = importlib.import_module('testing_intent_less_chatbot')

"""The retrievers were run multiple times. In each iteration the number of retrieved documents was increased
to find the optimal number of documents that needed to be returned in order to be able to answer the question."""
# Retrieve documents using the BERT retriever
# This loop was run seven times, varying the number of retrieved documents per question from 20 to 70,
# increasing by 10 each time.
l_results_BERT_retriever = []
l_retrieved_documents_content = []
l_references = []

for i, question in enumerate(questions):
    retrieved_documents = module.main(test='retriever', query=question,
                                      retriever=BERT_retriever(query=question, count=70,
                                                               index_path=testing_path + r'\testing_chatbot\indexes\BERT_index.json'))
    for doc in retrieved_documents:

        if doc['metadata']['source'] == 'webhelp_article':
            retrieved_document_content = doc['context']
            reference = doc['metadata']['Link']

        elif doc['metadata']['source'] == 'websupport_question':
            retrieved_document_content = doc['metadata']['answer']
            reference = doc['metadata']['question_id']

        l_retrieved_documents_content.append(retrieved_document_content)
        l_references.append(reference)

    dic_BERT_retriever = {
        'question': question,
        'question_ID': question_ids[i],
        'intent': intents[i],
        'retrieved_document_content': l_retrieved_documents_content,
        'reference': l_references,
        'human_response': human_responses[i],
        'source': 'BERT_retriever'
    }
    l_results_BERT_retriever.append(dic_BERT_retriever)
    l_retrieved_documents_content = []
    l_references = []

# Retrieve documents using the OpenAI retriever
# This loop was run five times, varying the number of retrieved documents per question from 10 to 50,
# increasing by 10 each time.
l_results_OpenAI_retriever = []
l_retrieved_documents_content = []
l_references = []
for i, question in enumerate(questions):
    retrieved_documents = module.main(test='retriever', query=question,
                                      retriever=OpenAI_retriever(query=question, count=70,
                                                                 index_path=testing_path + r'\testing_chatbot\indexes\OpenAI_index.json'))
    for doc in retrieved_documents:
        if doc['metadata']['source'] == 'webhelp_article':
            retrieved_document_content = doc['context']
            reference = doc['metadata']['Link']

        elif doc['metadata']['source'] == 'websupport_question':
            retrieved_document_content = doc['metadata']['answer']
            reference = doc['metadata']['question_id']

        l_retrieved_documents_content.append(retrieved_document_content)
        l_references.append(reference)

    dic_OpenAI_retriever = {
        'question': question,
        'question_ID': question_ids[i],
        'intent': intents[i],
        'retrieved_document_content': l_retrieved_documents_content,
        'reference': l_references,
        'human_response': human_responses[i],
        'source': 'OpenAI_retriever'
    }
    l_results_OpenAI_retriever.append(dic_OpenAI_retriever)
    l_retrieved_documents_content = []
    l_references = []

# 4. Save retrieved documents to csv file-------------------------------------------------------------------------------
# Retrieved documents were saved to csv file for subsequent evaluation.

combined_list = l_results_OpenAI_retriever + l_results_BERT_retriever

df = pd.DataFrame(combined_list,
                  columns=['question', 'question_ID', 'intent', 'retrieved_document_content', 'reference',
                           'human_response', 'source'])

# df.to_csv(testing_path + r'\testing_results\retriever_results\retriever_adjusted_scores_70.csv') # uncomment to resave

# 5. Find out how many times the correct document was retrieved---------------------------------------------------------
"""The core idea is to assess whether the retrieved documents are relevant to answer the question.
The result of this analysis is the number of answerable questions per retriever and number of retrieved documents.
A retrieval is considered successful, if at least one web support answer, that has the same intent as
the current question is retrieved or if one webhelp article, that is in the list of relevant web help articles for that
question is retrieved. If either of these conditions are fulfilled, it is assumed that the question can be answered with 
the retrieved documents"""

# This data set contains web support questions and answers. For each Q&A the intent and relevant webhelp articles are
# listed in separate columns
df_full_websupport_dataset = pd.read_csv(testing_path + r'\testing_data\cleaned_websupport_questions_with_intents_utf-8.csv')

# These csv files contain a sample of the websupport questions and a list of either 20, 30, 40, 50, 60 or 70
# retrieved documents (websupport questions and webhelp articles) per question.
# To see results without adjusted retriever scores, uncomment retriever results csv
csv_files_adjusted_scores = [
    (testing_path + r'\testing_results\retriever_results\retriever_adjusted_scores_20.csv', 20),
    (testing_path + r'\testing_results\retriever_results\retriever_adjusted_scores_30.csv', 30),
    (testing_path + r'\testing_results\retriever_results\retriever_adjusted_scores_40.csv', 40),
    (testing_path + r'\testing_results\retriever_results\retriever_adjusted_scores_50.csv', 50),
    (testing_path + r'\testing_results\retriever_results\retriever_adjusted_scores_60.csv', 60),
    (testing_path + r'\testing_results\retriever_results\retriever_adjusted_scores_70.csv', 70)
]

csv_files_unadjusted = [
    (testing_path + r'\testing_results\retriever_results\retriever_results_20.csv', 20),
    (testing_path + r'\testing_results\retriever_results\retriever_results_30.csv', 30),
    (testing_path + r'\testing_results\retriever_results\retriever_results_40.csv', 40),
    (testing_path + r'\testing_results\retriever_results\retriever_results_50.csv', 50),
    (testing_path + r'\testing_results\retriever_results\retriever_results_60.csv', 60),
    (testing_path + r'\testing_results\retriever_results\retriever_results_70.csv', 70)
]

# Create dictionary to understand which questions (identified by their IDs) are associated with which intents.
lists_by_intent = {}
for intent, group in df_full_websupport_dataset.groupby('intent'):
    lists_by_intent[intent] = group['Incident-ID'].tolist()

# Create dictionary to understand which questions (identified by their IDs) can be answered with which webhelp article.
lists_by_ID = {}
for ID, group in df_full_websupport_dataset.groupby('Incident-ID'):
    combined_webhelp_articles = group['webhelp article'].tolist() + group['webhelp article 2'].tolist()
    lists_by_ID[ID] = combined_webhelp_articles

# Initialize a dictionary to store the results
answerable_questions_adjusted_scores = {retriever: {num_docs: 0 for _, num_docs in csv_files_adjusted_scores} for retriever in ['OpenAI', 'DPR']}
unadjusted_answerable_questions = {retriever: {num_docs: 0 for _, num_docs in csv_files_unadjusted} for retriever in ['OpenAI', 'DPR']}

"""Iterate through each csv containing the retrieved documents per question and count how many of the questions can
be answered with the retrieved docs. THis is done once for the adjusted similarity scores and once for the unadjusted
scores"""
for csv_file, num_docs in csv_files_adjusted_scores:
    df = pd.read_csv(csv_file)

    for i, row in df.iterrows():
        count = 0
        question_intent = row['intent']  # Get intent of question
        # get list of retrieved web support question IDs and webhelp article links.
        retrieved_documents = ast.literal_eval(row['reference'])

        for doc in retrieved_documents:
            # Check if the retrieved doc has an ID that is associated with same intent as the question
            # for which it retrieved the doc.
            if doc in lists_by_intent.get(question_intent, []):
                count += 1
            # Check if a webhelp article that is associated with that question was retrieved.
            if doc in lists_by_ID.get(df_full_websupport_dataset['Incident-ID'][i], []):
                count += 1
        # If at least one web support answer with the same intent as question or one webhelp article that is assosiated
        # to the question is retrieved, the amount of answerable questions goes up by one.
        if count >= 1 and row['source'] == 'BERT_retriever':
            answerable_questions_adjusted_scores['DPR'][num_docs] += 1
        elif count >= 1 and row['source'] == 'OpenAI_retriever':
            answerable_questions_adjusted_scores['OpenAI'][num_docs] += 1

# Convert the counts of relevant retrievals to percentages.
# The percentage represents the percentage of answerable questions for the retrieved docs
total_questions = len(pd.read_csv(path + r'\testing\testing_results\retriever_results\retriever_adjusted_scores_20.csv'))/2
for retriever in answerable_questions_adjusted_scores:
    for num_docs in answerable_questions_adjusted_scores[retriever]:
        answerable_questions_adjusted_scores[retriever][num_docs] = (answerable_questions_adjusted_scores[retriever][num_docs] / total_questions) * 100

# Unadjusted Scores
for csv_file, num_docs in csv_files_unadjusted:
    df = pd.read_csv(csv_file)

    for i, row in df.iterrows():
        count = 0
        question_intent = row['intent']  # Get intent of question
        # get list of retrieved web support question IDs and webhelp article links.
        retrieved_documents = ast.literal_eval(row['reference'])

        for doc in retrieved_documents:
            # Check if the retrieved doc has an ID that is associated with same intent as the question
            # for which it retrieved the doc.
            if doc in lists_by_intent.get(question_intent, []):
                count += 1
            # Check if a webhelp article that is associated with that question was retrieved.
            if doc in lists_by_ID.get(df_full_websupport_dataset['Incident-ID'][i], []):
                count += 1
        # If at least one web support answer with the same intent as question or one webhelp article that is assosiated
        # to the question is retrieved, the amount of answerable questions goes up by one.
        if count >= 1 and row['source'] == 'BERT_retriever':
            unadjusted_answerable_questions['DPR'][num_docs] += 1
        elif count >= 1 and row['source'] == 'OpenAI_retriever':
            unadjusted_answerable_questions['OpenAI'][num_docs] += 1

# Convert the counts of relevant retrievals to percentages.
# The percentage represents the percentage of answerable questions for the retrieved docs
for retriever in unadjusted_answerable_questions:
    for num_docs in unadjusted_answerable_questions[retriever]:
        unadjusted_answerable_questions[retriever][num_docs] = (unadjusted_answerable_questions[retriever][num_docs] / total_questions) * 100

# 6. Visualizing the Answerability Rate---------------------------------------------------------------------------------
# Preparing data for plotting
labels = sorted(answerable_questions_adjusted_scores['OpenAI'].keys())
adjusted_DPR_counts = [answerable_questions_adjusted_scores['DPR'][label] for label in labels]
adjusted_openai_counts = [answerable_questions_adjusted_scores['OpenAI'][label] for label in labels]
unadjusted_DPR_counts = [unadjusted_answerable_questions['DPR'][label] for label in labels]
unadjusted_openai_counts = [unadjusted_answerable_questions['OpenAI'][label] for label in labels]

x = np.arange(len(labels))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots()

# Plot the adjusted results
rects1 = ax.bar(x - width / 2, adjusted_DPR_counts, width, label='Adjusted BERT Retriever', color='#1a2639')
rects2 = ax.bar(x + width / 2, adjusted_openai_counts, width, label='Adjusted OpenAI Retriever', color='#c24d2c')

# Plot the unadjusted results
rects3 = ax.bar(x - width / 2, unadjusted_DPR_counts, width, label='Unadjusted BERT Retriever', alpha=0.5, color='#1f77b4', hatch='//')
rects4 = ax.bar(x + width / 2, unadjusted_openai_counts, width, label='Unadjusted OpenAI Retriever', alpha=0.5, color='#ff7f0e', hatch='//')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.tick_params(axis='y', labelsize=14)
ax.set_xlabel('Number of Retrieved Documents per Question', fontsize=16)
ax.set_ylabel('Question Answerability Rate', fontsize=16)
#ax.set_title('Question Answerability Rate by Document Count and Retriever', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)
ax.set_yticks(np.arange(0, max(adjusted_DPR_counts + adjusted_openai_counts + unadjusted_DPR_counts + unadjusted_openai_counts) + 10, 10))

# Adding customized grid lines
ax.grid(True, color='#D9D9D9')  # Set the color of the grid lines
ax.set_axisbelow(True)

ax.legend(fontsize=14)

# Adjust the figure size to make room for the legend
fig.set_size_inches(12, 6)
# Place the legend to the right of the plot
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
# Adjust the layout to take the new legend location into account
plt.tight_layout()

plt.show()


# Print the answerability rates for each retriever and number of retrieved documents
print('Answerability Rates: ' + str(unadjusted_answerable_questions))
for retriever, docs_counts in unadjusted_answerable_questions.items():
    print(docs_counts)
    print(f'\nRetriever: {retriever}')
    for num_docs, rate in docs_counts.items():
        print(f'  Documents Retrieved: {num_docs}, Answerability Rate: {rate:.2f}%')

# 7. Calculating precision, recall and F1 score of retrievers-----------------------------------------------------------
with open(path + r"\testing\testing_chatbot\indexes\OpenAI_index.json", 'r', encoding='utf-8') as file:
    index = json.load(file)

intent_count = {}

# Create Dictionary with each intent as key and number of occurences as values
for element in index:
    if element['metadata']['source'] == "websupport_question":
    # Extract the intent from the metadata
        intent = element['metadata']['intent']

        # Update the count of the intent in the dictionary
        if intent in intent_count:
            intent_count[intent] += 1
        else:
            intent_count[intent] = 1


# Initialize a dictionary to store precision, recall and f1 score per retriever and number of retrieved documents
results = {}

for csv_file, num_docs in csv_files_adjusted_scores:
    results[num_docs] = {'OpenAI_retriever': {'precision': 0, 'recall': 0, 'count': 0},
                         'BERT_retriever': {'precision': 0, 'recall': 0, 'count': 0}}

    df = pd.read_csv(csv_file)

    for i, row in df.iterrows():
        total_number_relevant_docs = 0
        question_intent = row['intent']  # Get intent of question
        # get list of retrieved web support question IDs and webhelp article links.
        retrieved_documents = ast.literal_eval(row['reference'])
        total_number_of_docs = len(retrieved_documents)

        for doc in retrieved_documents:
            # Check if the retrieved doc has an ID that is associated with same intent as the question
            # for which it retrieved the doc.
            if doc in lists_by_intent.get(question_intent, []):
                total_number_relevant_docs += 1
            # Check if a webhelp article that is associated with that question was retrieved.
            if doc in lists_by_ID.get(df_full_websupport_dataset["Incident-ID"][i], []):
                total_number_relevant_docs += 1

        # Calculate precision and recall
        if total_number_of_docs > 0 and question_intent in intent_count:
            precision = total_number_relevant_docs / total_number_of_docs
            recall = total_number_relevant_docs / intent_count[question_intent]

            retriever = row['source']
            if retriever in results[num_docs]:
                results[num_docs][retriever]['precision'] += precision
                results[num_docs][retriever]['recall'] += recall
                results[num_docs][retriever]['count'] += 1

# Calculate the average for each retriever at each doc count
for num_docs in results:
    for retriever in results[num_docs]:
        if results[num_docs][retriever]['count'] > 0:
            results[num_docs][retriever]['precision'] /= results[num_docs][retriever]['count']
            results[num_docs][retriever]['recall'] /= results[num_docs][retriever]['count']

        # If at least one web support answer with the same intent as question or one webhelp article that is assosiated
        # to the question is retrieved, the amount of answerable questions goes up by one.
        if row['source'] == "BERT_retriever":
            precision_bert = total_number_relevant_docs/total_number_of_docs
            recall = total_number_relevant_docs / intent_count[question_intent]
        elif row['source'] == "OpenAI_retriever":
            precision_openAI = total_number_relevant_docs/total_number_of_docs
            recall = total_number_relevant_docs / intent_count[question_intent]

# Iterate through the dictionary and calculate F1 scores
for num_docs, retrievers in results.items():
    for retriever, metrics in retrievers.items():
        precision = metrics['precision']
        recall = metrics['recall']
        # Calculate F1 score, avoid division by zero
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        # Add F1 score to the dictionary
        metrics['f1_score'] = f1_score

print(results)
# Choose a beta value greater than 1 to emphasize recall
beta = 2  # for example, you can adjust this as needed

# Iterate through the dictionary and calculate Fβ scores
for num_docs, retrievers in results.items():
    for retriever, metrics in retrievers.items():
        precision = metrics['precision']
        recall = metrics['recall']
        # Calculate Fβ score, avoid division by zero
        if (beta**2 * precision) + recall > 0:
            f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        else:
            f_beta = 0
        # Add Fβ score to the dictionary
        metrics['f_beta'] = f_beta

# 8. Visualizing precision, recall and F1 score-------------------------------------------------------------------------
# Parameters for plotting
num_docs_list = [20, 30, 40, 50, 60, 70]
metrics = ['precision', 'recall', 'f1_score']
retrievers = ['OpenAI_retriever', 'BERT_retriever']
colors = ['#c24d2c', '#3e4a61', '#1a2639']  # Different colors for each metric
bar_width = 0.35
index = np.arange(len(retrievers))

# Create subplots
fig, axes = plt.subplots(1, 6, figsize=(20, 6), sharey=True)

for k, num_docs in enumerate(num_docs_list):
    for i, metric in enumerate(metrics):
        values = [results.get(num_docs, {}).get(retriever, {}).get(metric, 0) for retriever in retrievers]
        axes[k].bar(index + bar_width*i, values, bar_width, color=colors[i], label=metric)

    axes[k].set_title(f'{num_docs} Docs')
    axes[k].set_xticks(index + bar_width)
    axes[k].set_xticklabels(retrievers, rotation=45)
    if k == 0:
        axes[k].set_ylabel('Scores')
    axes[k].set_ylim(0, 1)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(metrics), bbox_to_anchor=(0.5, 0.05))

# Adjust the layout to make room for the legend
plt.tight_layout(rect=[0, 0.1, 1, 0.9])
# plt.suptitle('Comparison of Precision, Recall, and F1 Score for OpenAI and BERT Retriever')
plt.show()