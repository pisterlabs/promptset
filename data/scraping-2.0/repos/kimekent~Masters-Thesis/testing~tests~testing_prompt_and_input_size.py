"""
This script is designed to analyse how different prompts and input sizes affect the quality of generated
chatbot responses.

It encompasses several key stages:

- Response Generation: Utilizing OpenAIs gpt-3.5-turbo-1106 model to generate responses to a set of
    web support questions. The prompt and number of documents used to answer the question are varied (step 2).

- Performance Evaluation: The generated responses are evaluated against human responses using BLEU-4, ROUGE-L,
    and semantic similarity. These metrics are calculated for each combination of prompt and input size (step 4).

- Visualization: Results of the performance evaluation are visualized using line charts (step 5).

To run this script update the 'path' variable to the root directory of this project and add your OpenAI API key to
'openaiapikey.txt' in the root directory of this project.

To run the evaluation on the generated responses execute steps 3-6.
"""


# 1. Setup--------------------------------------------------------------------------------------------------------------
# Set path to root directory and OpenAI API key
import sys
path = r'C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis' # Change
testing_path = path + r'\testing'
sys.path.append(testing_path)

from testing_chatbot.testing_functions import open_file
import openai
import os
os.environ['OPENAI_API_KEY'] = open_file(path + '\openaiapikey.txt')
openai.api_key = os.getenv('OPENAI_API_KEY') # Add OpenAI API key to this .txt file

# Standard Libraries
import os
import glob
import pandas as pd
import csv

# Libraries for running the testing chatbot / for text generation
import importlib
from testing_chatbot.testing_functions import OpenAI_retriever, replace_links_with_placeholder

# Library run BLEU, ROUGE and BERTScore tests
from evaluate import load
from nltk.translate.bleu_score import corpus_bleu

# Library for visualizing results
import matplotlib.pyplot as plt
import ast

# Import testing dataset
testing_data_set = pd.read_csv(testing_path + r'\testing_data\test_dataset.csv')
questions = testing_data_set['Beschreibung'].tolist()
human_responses = testing_data_set['Lösungsbeschreibung'].tolist()
"""Links in the original data were non-informative and replaced with placeholders during index creation.
To ensure consistency, the same transformation is applied to these questions, as they will be used
for index comparison."""
questions = [replace_links_with_placeholder(i) for i in questions]

# Import OpenAI testing chatbot
"""This testing chatbot is specifically designed for evaluating individual chatbot components.
It accepts various parameters, including the test type (retriever, prompt_max_input, memory), the selected retriever,
the chosen prompt, and the number of document tokens to add to the prompt, to systematically assess the chatbot's
performance across these different variables."""

sys.path.append(testing_path + r'\testing_chatbot')
module = importlib.import_module('testing_intent_less_chatbot')


# 2. Generate answers with different prompts and input sizes------------------------------------------------------------
# Create dictionary with all prompts in the prompt folder. Prompt name is the key. Prompt text is the value.
# These are these are the prompts that will be tested against each other
os.chdir(os.path.join(testing_path,"testing_chatbot", "testing_prompts", "answer_generation"))
d_prompts = {
    "_".join(file.split("_")[1:]).split(".txt")[0]: open_file(os.path.join(testing_path, "testing_chatbot", "testing_prompts",
                                                                           "answer_generation", file))
    for file in glob.glob("*.txt")}

# Initialize lists to store generated answers per prompt and input size
generated_responses = []
final_results_list = []

"""In the following loop, each combination of prompt and input size is used to generate an answer for all questions
in the testing dataset."""

for prompt_name, prompt_content in d_prompts.items():
    # current_max_input is the number of tokens that the question prompt will use.
    # It is iteratively increased to find the optimum value.
    current_max_input = 6000
    # Maximum input value is set to 14000 to leave enough tokens for response generation and avoid token limit errors.
    while current_max_input <= 14000:
        generated_responses = []  # Reset list for each input increment

        for i, question in enumerate(questions):
            response, websupport_question_id, l_websupport_questions, l_webhelp_link, l_webhelp_articles = module.main(
                test="prompt_max_input", query=question,
                retriever=OpenAI_retriever(query=question,
                                           index_path=testing_path + r"\testing_chatbot\indexes\OpenAI_index.json"),
                question_prompt=prompt_content, max_input=current_max_input)
            generated_responses.append(response)

        result_dict = {
            "prompt": prompt_name,
            "max_input": current_max_input,
            "generated_responses": generated_responses
        }

        final_results_list.append(result_dict)
        current_max_input += 2000 # increase token limit for question prompt.


# 3. Save output of generation to csv for future analysis---------------------------------------------------------------
df = pd.DataFrame(final_results_list)
# df.to_csv(testing_path + r"\testing_results\prompt_and_max_input_results.csv") # uncomment to resave


# 4. Calculate BLEU-4, ROUGE-L and semantic similarity-------------------------------------------------------------

rouge = load("rouge")
bertscore = load("bertscore")

# Get human responses to compare to AI generated responses
testing_data_set = pd.read_csv(testing_path + r"\testing_data\test_dataset.csv")
human_responses = testing_data_set["Lösungsbeschreibung"].tolist()

def tokenizer(reference, candidate):
    tokenized_candidate = [answer.split() for answer in candidate]
    tokenized_reference = [[answer.split()] for answer in reference]
    return tokenized_reference, tokenized_candidate

# Initialize dictionary to store scores per prompt and input size
bleu_scores = {}
rouge_scores = {}
semantic_similarity = {}

# Loop that calculates corpus BLEU, ROUGE-L and semantic similarity scores for each prompt and input size combination---
with open(testing_path + r"\testing_results\prompt_and_max_input_results.csv", mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)  # Skip the header row

    for row in csv_reader:
        prompt = row[1]
        max_input = str(row[2])
        generated_responses = ast.literal_eval(row[3]) # Convert string to list
        tokenized_reference, tokenized_candidate = tokenizer(human_responses, generated_responses)
        bleu_result = corpus_bleu(tokenized_reference, tokenized_candidate)
        rouge_results = rouge.compute(predictions=generated_responses, references=human_responses)
        semantic_similarity_results = bertscore.compute(predictions=generated_responses, references=human_responses, lang="de")
        if prompt not in bleu_scores:
            bleu_scores[prompt] = {}
        bleu_scores[prompt][max_input] = bleu_result
        if prompt not in rouge_scores:
            rouge_scores[prompt] = {}
        rouge_scores[prompt][max_input] = rouge_results
        if prompt not in semantic_similarity:
            semantic_similarity[prompt] = {}
        semantic_similarity[prompt][max_input] = semantic_similarity_results


# Loop that calculates corpus BLEU, ROUGE-L, and semantic similarity scores for each prompt and input size combination
for prompt_type, scores in bleu_scores.items():
    print(f"\nBLEU Scores for {prompt_type}:")
    for input_size, bleu_score in scores.items():
        print(f"  Input Size: {input_size}, BLEU Score: {bleu_score}")

for prompt_type, scores in rouge_scores.items():
    print(f"\nROUGE Scores for {prompt_type}:")
    for input_size, rouge_score in scores.items():
        print(f"  Input Size: {input_size}, ROUGE-L Score: {rouge_score['rougeL']}")

for prompt_type, scores in semantic_similarity.items():
    print(f"\nSemantic Similarity Scores for {prompt_type}:")
    for input_size, sim_score in scores.items():
        avg_f1_score = sum(sim_score['f1']) / len(sim_score['f1'])
        print(f"  Input Size: {input_size}, Average F1-Score: {avg_f1_score}")


# 5. Result Visualization-----------------------------------------------------------------------------------------------

# Colors used in line chart
color_map = {
    "original_prompt": "#1a2639",
    "avoiding_repetition_in_output": "#448ef6",
    "avoid_hallucination": "#c24d2c",
}

# Legend labels
legend_labels = {
    'original_prompt': 'Original Prompt',
    'avoiding_repetition_in_output': 'Avoid Repetition Prompt',
    'avoid_hallucination': 'Avoid Hallucination Prompt'
}

# Plotting the data
plt.figure(figsize=(12, 6))

# Visualizing corpus BLEU scores
for prompt_type, scores in bleu_scores.items():
    input_sizes = [int(size) for size in scores.keys()]
    bleu_values = [score for score in scores.values()]
    plt.plot(input_sizes, bleu_values, label=legend_labels[prompt_type], color=color_map[prompt_type])

    # Highlight points at specific input sizes
    highlight_sizes = [6000, 8000, 10000, 12000, 14000]
    # Extract the BLEU values for the specific input sizes
    highlight_values = [scores[str(size)] for size in highlight_sizes if str(size) in scores]
    plt.scatter(highlight_sizes, highlight_values, color=color_map[prompt_type], zorder=5)

plt.xticks([6000, 8000, 10000, 12000, 14000])
plt.xlim(6000, 14000)
plt.ylim(0, 1)

plt.xlabel('Maximum Number of Input Tokens', fontsize=14)
plt.ylabel('BLEU-4 Score', fontsize=14)
# plt.title('BLEU-4 Scores by Maximum Number of Input Tokens for Different Prompts', fontsize=20)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(True)
plt.subplots_adjust(right=0.75)
plt.show()

# Visualizing ROUGE-L scores
for prompt_type, scores in rouge_scores.items():
    input_sizes = [int(size) for size in scores.keys()]
    rouge_values = [i["rougeL"] for i in scores.values()]
    plt.plot(input_sizes, rouge_values, label=legend_labels[prompt_type], color=color_map[prompt_type])

    # Highlight points at specific input sizes
    highlight_sizes = [6000, 8000, 10000, 12000, 14000]
    # Extract the ROUGE-L values for the specific input sizes
    highlight_values = [scores[str(size)]["rougeL"] for size in highlight_sizes if str(size) in scores]
    plt.scatter(highlight_sizes, highlight_values, color=color_map[prompt_type], zorder=5)

plt.xticks([6000, 8000, 10000, 12000, 14000])
plt.xlim(6000, 14000)
plt.ylim(0, 1)


plt.xlabel('Maximum Number of Input Tokens', fontsize=14)
plt.ylabel('ROUGE-L Score', fontsize=14)
# plt.title('Rouge-L Scores by Maximum Number of Input Tokens for Different Prompts', fontsize=20)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(True)
plt.subplots_adjust(right=0.75)
plt.show()


# Visualizing BERTscore
for prompt_type, scores in semantic_similarity.items():
    input_sizes = [int(size) for size in scores.keys()]
    print(input_sizes)
    semantic_similarity_scores = [sum(i["f1"])/len(i["f1"]) for i in scores.values()]
    print(semantic_similarity_scores)
    plt.plot(input_sizes, semantic_similarity_scores, label=legend_labels[prompt_type], color=color_map[prompt_type])

    # Add dots at specific input sizes
    highlight_sizes = [6000, 8000, 10000, 12000, 14000]
    highlight_values = [
        sum(scores[str(size)]["f1"]) / len(scores[str(size)]["f1"])
        for size in highlight_sizes if str(size) in scores
    ]
    plt.scatter(highlight_sizes, highlight_values, color=color_map[prompt_type], zorder=5)

plt.xticks([6000, 8000, 10000, 12000, 14000])
plt.xlim(6000, 14000)
plt.ylim(0, 1)

plt.xlabel('Maximum Number of Input Tokens', fontsize=14)
plt.ylabel('Average F1-Score over all Generated Answers ', fontsize=14)
# plt.title('Semantic Similarity Scores by Maximum Number of Input Tokens for Different Prompts', fontsize=20)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(True)
plt.subplots_adjust(right=0.75)

plt.show()
#-----------------------------------------------------------------------------------------------------------------------