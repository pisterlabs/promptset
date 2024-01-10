"""
This script is designed for evaluating the memory component of the intent-less chatbot, with a particular focus on its
ability to handle conversation turns effectively. Key aspects of the script include:

- Chatbot Testing: The script tests the chatbot by using a dataset that contains conversation turns.
    During this process, the chatbot may reformulate the question to improve document retrieval accuracy
     before generating a response (Step 2).

- Performance Evaluation: The script quantitatively assesses the chatbot's performance using BLEU-4, ROUGE-L,
    and BERTScore by comparing the chatbot-generated responses to those from humans (Step 4).

- Visualization: Results are visualized with bar charts to provide a clear comparison of performance metrics (Step 5).

To run this script update the 'path' variable to the root directory of this project and add your OpenAI API key to
'openaiapikey.txt' in the root directory of this project.

To run the evaluation on the generated responses execute steps 4-5.
"""


# 1. Set up-------------------------------------------------------------------------------------------------------------
# Set path to root directory and OpenAI API key
import sys
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis" # Change
testing_path = path + r'\testing'
sys.path.append(testing_path)

from testing_chatbot.testing_functions import open_file
import openai
import os
os.environ['OPENAI_API_KEY'] = open_file(path + '\openaiapikey.txt')
openai.api_key = os.getenv('OPENAI_API_KEY') # Add OpenAI API key to this .txt file

# Import Libraries
# Standard Libraries
import pandas as pd

# Libraries for running the testing chatbot / for text generation
import importlib
from testing_chatbot.testing_functions import OpenAI_retriever

# Libraries to evaluate the generated responses
from evaluate import load
from nltk.translate.bleu_score import corpus_bleu

# Libraries for visualizing the results
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Import testing chatbot to test memory component
"""This testing chatbot is specifically designed for evaluating individual chatbot components.
It accepts various parameters, including the test type (retriever, prompt_max_input, memory), the selected retriever,
the chosen prompt, and the number of document tokens to add to the prompt, to systematically assess the chatbot's
performance across these different variables.
The memory component of the test bot evaluates how well the chatbot can handle conversation turns by utilizing
information stored in memory."""

sys.path.append(testing_path + r'\testing_chatbot')
module = importlib.import_module('testing_intent_less_chatbot')

# Import testing dataset: This dataset contains a collection of conversations. Each conversation is
# made up of a sequence of interrelated questions, where each question refers to something from the
# previous question.

df_questions_set = pd.read_csv(testing_path + r'\testing_data\conversation_turns.csv')

# Initialize dictionaries
d_conversation_list = {} # Stores list of questions of each conversation
d_human_reformulated_questions = {} # Stores the human reference reformulated questions
d_human_answers = {} # Stores the human reference answers

# Add data to the dictionaries from the conversation turn dataset
for column_name in df_questions_set.columns:
    if 'questions' in column_name: # Check if the column name contains 'questions'
        d_conversation_list[column_name] = df_questions_set[column_name].tolist() # Add column values to the dictionary

for column_name in df_questions_set.columns:
    if 'human_reformulated' in column_name: # Check if the column name contains 'human_reformulated'
        # Add column values to the dictionary
        d_human_reformulated_questions[column_name] = df_questions_set[column_name].tolist()

for column_name in df_questions_set.columns:
    if 'human_answer' in column_name: # Check if the column name contains 'human_answer'
        d_human_answers[column_name] = df_questions_set[column_name].tolist() # Add column values to the dictionary

# Delete 'nan' values in conversation list
for key, value_list in d_conversation_list.items():
    d_conversation_list[key] = [value for value in value_list if not pd.isna(value)]
for key, value_list in d_human_reformulated_questions.items():
    d_human_reformulated_questions[key] = [value for value in value_list if not pd.isna(value)]
for key, value_list in d_human_answers.items():
    d_human_answers[key] = [value for value in value_list if not pd.isna(value)]

l_human_reformulated_questions = list(d_human_reformulated_questions.values())
l_human_answers = list(d_human_answers.values())

# Set prompt paths of all the retriever paths that need to be tested
d_prompt_paths = {'original_prompt': testing_path + r'\testing_chatbot\testing_prompts\retriever\original_retriever_prompt.txt',
                  'few_shot_prompt': testing_path + r'\testing_chatbot\testing_prompts\retriever\few_shot_prompt.txt',
                  'instruction_prompt': testing_path + r'\testing_chatbot\testing_prompts\retriever\instruction_based_prompting.txt'}

# Initialize lists to store returned results
row = []
data = []
i = 0


# 2. Run chatbot on all conversations using different retriever prompts------------------------------------------------

# The goal of the retriever prompt is to reformulate questions that refer to earlier conversation turns.
for retriever_prompt_name, retriever_prompt_path in d_prompt_paths.items():
    for conversation_name, l_questions in d_conversation_list.items():
        # value is a list of questions, were each question refers to the previous. One list represents one conversation.
        l_questions = l_questions
        l_generated_responses, l_restructured_queries = module.main(test='memory',
                                                                    query=l_questions,
                                                                    retriever=OpenAI_retriever(query='placeholder',
                                                                                               index_path=testing_path + r'\testing_chatbot\indexes\OpenAI_index.json'),
                                                                    retriever_prompt=retriever_prompt_path,
                                                                    question_prompt=testing_path + r'\testing_chatbot\testing_prompts\answer_generation_with_history\qa_prompt_with_history.txt',
                                                                    max_input=12000)

        row.append(retriever_prompt_name)
        row.append(conversation_name)
        row.append(l_questions)
        row.append(l_restructured_queries)
        row.append(l_human_reformulated_questions[i])
        row.append(l_generated_responses)
        row.append(l_human_answers[i])
        data.append(row)
        row = []
        i += 1

    i = 0


# 3. Save output of generation to csv for further analysis--------------------------------------------------------------
df = pd.DataFrame(data, columns=['prompt_name', 'conversation_number', 'questions',
                                 'ai_reformulated_question', 'human_reformulated_question',
                                 'generated_answer', 'human_answers'])

# df.to_csv(testing_path + r'\testing_results\memory_results.csv') # uncomment to resave

# 4. Evaluate memory component------------------------------------------------------------------------------------------
df = pd.read_csv(testing_path + r'\testing_results\memory_results.csv')

# Convert string representations of lists to actual lists. When saving to csv lists were converted to strings.
df['human_reformulated_question'] = df['human_reformulated_question'].apply(lambda x: x[2:-2].split("', '"))
df['ai_reformulated_question'] = df['ai_reformulated_question'].apply(lambda x: x[2:-2].split(", '"))
df['generated_answer'] = df['generated_answer'].apply(lambda x: x[2:-2].split(", '"))
df['human_answers'] = df['human_answers'].apply(lambda x: x[2:-2].split(", '"))

# Human reformulated question and human answers is the same for each prompt.
# They are what the generated reformulated question and generated answer will be compared to.
human_reformulate = sum(df[df['prompt_name'] == 'original_prompt']['human_reformulated_question'], [])
human_answers = sum(df[df['prompt_name'] == 'original_prompt']['human_answers'], [])

# Store all results for reformulation task in a list per prompt
original_prompt_ai_reformulate = sum(df[df['prompt_name'] == 'original_prompt']['ai_reformulated_question'], [])
few_shot_prompt_ai_reformulate = sum(df[df['prompt_name'] == 'few_shot_prompt']['ai_reformulated_question'], [])
instruction_prompt_ai_reformulate = sum(df[df['prompt_name'] == 'instruction_prompt']['ai_reformulated_question'], [])

# Store all results of answer generation in a list per prompt
original_prompt_generated_answer = sum(df[df['prompt_name'] == 'original_prompt']['generated_answer'], [])
few_shot_prompt_generated_answer = sum(df[df['prompt_name'] == 'few_shot_prompt']['generated_answer'], [])
instruction_prompt_generated_answer = sum(df[df['prompt_name'] == 'instruction_prompt']['generated_answer'],[])

# Calculate BLEU-4 score for question reformulation and response generation for each prompt
human_reformulate_tokenized = [[question.split()] for question in human_reformulate]
human_answers_tokenized = [[question.split()] for question in human_answers]

original_prompt_ai_reformulate_nested_list = [question.split() for question in original_prompt_ai_reformulate]
corpus_bleu_original_prompt_reformulate = corpus_bleu(human_reformulate_tokenized, original_prompt_ai_reformulate_nested_list)

few_shot_prompt_ai_reformulate_nested_list = [question.split() for question in few_shot_prompt_ai_reformulate]
corpus_bleu_few_shot_prompt_reformulate = corpus_bleu(human_reformulate_tokenized, few_shot_prompt_ai_reformulate_nested_list)
instruction_prompt_ai_reformulate_nested_list = [question.split() for question in instruction_prompt_ai_reformulate]
corpus_bleu_instruction_prompt_reformulate = corpus_bleu(human_reformulate_tokenized, instruction_prompt_ai_reformulate_nested_list)

original_prompt_generated_answers_nested_list = [question.split() for question in original_prompt_generated_answer]
corpus_bleu_original_prompt_generated_answer = corpus_bleu(human_answers_tokenized, original_prompt_generated_answers_nested_list)

few_shot_prompt_generated_answer_nested_list = [question.split() for question in few_shot_prompt_generated_answer]
corpus_bleu_few_shot_prompt_generated_answer = corpus_bleu(human_answers_tokenized, few_shot_prompt_generated_answer_nested_list)

instruction_prompt_generated_answer_nested_list = [question.split() for question in instruction_prompt_generated_answer]
corpus_bleu_instruction_prompt_generated_answer = corpus_bleu(human_answers_tokenized, instruction_prompt_generated_answer_nested_list)

# Calculate the ROUGE scores for question reformulation and response generation for each prompt
rouge = load('rouge')
rouge_original_prompt_question_reformulate = rouge.compute(predictions=original_prompt_ai_reformulate,
                                                           references=human_reformulate)
rouge_few_shot_prompt_question_reformulate = rouge.compute(predictions=few_shot_prompt_ai_reformulate,
                                                           references=human_reformulate)
rouge_instruction_prompt_question_reformulate = rouge.compute(predictions=instruction_prompt_ai_reformulate,
                                                              references=human_reformulate)

rouge_original_prompt_generated_response = rouge.compute(predictions=original_prompt_generated_answer,
                                                         references=human_answers)
rouge_few_shot_prompt_generated_response = rouge.compute(predictions=few_shot_prompt_generated_answer,
                                                         references=human_answers)
rouge_instruction_prompt_generated_response = rouge.compute(predictions=instruction_prompt_generated_answer,
                                                            references=human_answers)

# Calculate semantic similarity for question reformulation and response generation for each prompt
bertscore = load('bertscore')
bertscore_original_prompt_question_reformulate = bertscore.compute(predictions=original_prompt_ai_reformulate,
                                                                   references=human_reformulate, lang='de')
bertscore_few_shot_prompt_question_reformulate = bertscore.compute(predictions=few_shot_prompt_ai_reformulate,
                                                                   references=human_reformulate, lang='de')
bertscore_instruction_prompt_question_reformulate = bertscore.compute(predictions=instruction_prompt_ai_reformulate,
                                                                      references=human_reformulate, lang='de')

bertscore_original_prompt_generated_response = bertscore.compute(predictions=original_prompt_generated_answer,
                                                                 references=human_answers, lang='de')
bertscore_few_shot_prompt_generated_response = bertscore.compute(predictions=few_shot_prompt_generated_answer,
                                                                 references=human_answers, lang='de')
bertscore_instruction_prompt_generated_response = bertscore.compute(predictions=instruction_prompt_generated_answer,
                                                                    references=human_answers, lang='de')


# 5. Visualize results for each prompt for response generation and question restructuring-------------------------------

tasks = ['Question Reformulation', 'Answer Generation']
prompts = ['Original Prompt', 'Few-Shot Prompt', 'Instruction Prompt']

# Grouping the results by prompt

# BLEU scores
bleu_results_original = [corpus_bleu_original_prompt_reformulate,
                         corpus_bleu_original_prompt_generated_answer]
bleu_results_few_shot = [corpus_bleu_few_shot_prompt_reformulate,
                         corpus_bleu_few_shot_prompt_generated_answer]
bleu_results_instruction = [corpus_bleu_instruction_prompt_reformulate,
                            corpus_bleu_instruction_prompt_generated_answer]
# ROUGE scores
rouge_results_original = [rouge_original_prompt_question_reformulate['rougeL'],
                          rouge_original_prompt_generated_response['rougeL']]
rouge_results_few_shot = [rouge_few_shot_prompt_question_reformulate['rougeL'],
                          rouge_few_shot_prompt_generated_response['rougeL']]
rouge_results_instruction = [rouge_instruction_prompt_question_reformulate['rougeL'],
                             rouge_instruction_prompt_generated_response['rougeL']]

# Semantic similarity score
# The output of the BERTSimilarity Score is a list of F1-scores (one score per generated and human answer)
# In the bar chart the F1-score averaged over all answers per prompt and task combination is plotted.

# Calculate average F1-score per prompt and task
average_f1_original_prompt_reformulate = sum(
    bertscore_original_prompt_question_reformulate['f1']) / len(
    bertscore_original_prompt_question_reformulate['f1'])
average_f1_original_prompt_generated_response = sum(
    bertscore_original_prompt_generated_response['f1']) / len(
    bertscore_original_prompt_generated_response['f1'])

average_f1_few_shot_prompt_reformulate = sum(
    bertscore_few_shot_prompt_question_reformulate['f1']) / len(
    bertscore_few_shot_prompt_question_reformulate['f1'])
average_f1_few_shot_prompt_generated_response = sum(
    bertscore_few_shot_prompt_generated_response['f1']) / len(
    bertscore_few_shot_prompt_generated_response['f1'])

average_f1_instruction_prompt_reformulate = sum(
    bertscore_instruction_prompt_question_reformulate['f1']) / len(
    bertscore_instruction_prompt_question_reformulate['f1'])
average_f1_instruction_prompt_generated_response = sum(
    bertscore_instruction_prompt_generated_response['f1']) / len(
    bertscore_instruction_prompt_generated_response['f1'])

# Group average f1 score by prompt for plotting
s_similarity_score_results_original = [average_f1_original_prompt_reformulate,
                                       average_f1_original_prompt_generated_response]
s_similarity_score_results_few_shot = [average_f1_few_shot_prompt_reformulate,
                                       average_f1_few_shot_prompt_generated_response]
s_similarity_score_results_instruction = [average_f1_instruction_prompt_reformulate,
                                          average_f1_instruction_prompt_generated_response]

bar_width = 0.2
index = np.arange(len(tasks))

# Create BLEU bar chart
fig, ax = plt.subplots()
bar1 = ax.bar(index, bleu_results_original, bar_width, label='Original Prompt', color='#1a2639')
bar2 = ax.bar(index + bar_width, bleu_results_few_shot, bar_width, label='Few-Shot Prompt', color='#3e4a61')
bar3 = ax.bar(index + 2 * bar_width, bleu_results_instruction, bar_width, label='Instruction Prompt', color='#c24d2c')

# Customization BLEU bar chart
ax.set_xlabel('Tasks', fontsize=16)
ax.set_ylabel('BLEU-4 Score', fontsize=16)
#ax.set_title('BLEU-4 Scores by Retriever Prompt and Task', fontsize=20)
ax.set_xticks(index + bar_width)
ax.set_xticklabels(tasks, fontsize=14)
ax.legend(fontsize=12)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_ylim([0, 1])

# Adding customized grid lines
ax.grid(True, color='#D9D9D9')  # Set the color of the grid lines
ax.set_axisbelow(True)

# Show BLEU plot
plt.show()

# Create  ROUGE bar chart
index = np.arange(len(tasks))
fig, ax = plt.subplots()
bar1 = ax.bar(index, rouge_results_original, bar_width, label='Original Prompt', color='#1a2639')
bar2 = ax.bar(index + bar_width, rouge_results_few_shot, bar_width, label='Few-Shot Prompt', color='#3e4a61')
bar3 = ax.bar(index + 2 * bar_width, rouge_results_instruction, bar_width, label='Instruction Prompt', color='#c24d2c')

# Customize ROUGE bar chart
ax.set_xlabel('Tasks', fontsize=16)
ax.set_ylabel('ROUGE-L Score', fontsize=16)
#ax.set_title('ROUGE-L Scores by Retriever Prompt and Task', fontsize=20)
ax.set_xticks(index + bar_width)
ax.set_xticklabels(tasks, fontsize=14)
ax.legend(fontsize=12)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_ylim([0, 1])

# Adding customized grid lines
ax.grid(True, color='#D9D9D9')  # Set the color of the grid lines
ax.set_axisbelow(True)

# Show ROUGE plot
plt.show()

# Create semantic similarity bar chart
index = np.arange(len(tasks))
fig, ax = plt.subplots()
bar1 = ax.bar(index, s_similarity_score_results_original, bar_width, label='Original Prompt', color='#1a2639')
bar2 = ax.bar(index + bar_width, s_similarity_score_results_few_shot, bar_width, label='Few-Shot Prompt',
              color='#3e4a61')
bar3 = ax.bar(index + 2 * bar_width, s_similarity_score_results_instruction, bar_width, label='Instruction Prompt',
              color='#c24d2c')

# Customize semantic similarity bar chart
ax.set_xlabel('Tasks', fontsize=16)
ax.set_ylabel('Average F1-Score', fontsize=16)
#ax.set_title('Semantic Similarity Score by Retriever Prompt and Task', fontsize=20)
ax.set_xticks(index + bar_width)
ax.set_xticklabels(tasks, fontsize=14)
ax.legend(fontsize=12)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_ylim([0, 1])

# Adding customized grid lines
ax.grid(True, color='#D9D9D9')  # Set the color of the grid lines
ax.set_axisbelow(True)

# Show semantic similarity bar plot
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
# Print BLEU Scores
print('BLEU Scores:')
print(f'Original Prompt Reformulation: {corpus_bleu_original_prompt_reformulate:.4f}')
print(f'Few-Shot Prompt Reformulation: {corpus_bleu_few_shot_prompt_reformulate:.4f}')
print(f'Instruction Prompt Reformulation: {corpus_bleu_instruction_prompt_reformulate:.4f}')
print(f'Original Prompt Generated Answer: {corpus_bleu_original_prompt_generated_answer:.4f}')
print(f'Few-Shot Prompt Generated Answer: {corpus_bleu_few_shot_prompt_generated_answer:.4f}')
print(f'Instruction Prompt Generated Answer: {corpus_bleu_instruction_prompt_generated_answer:.4f}\n')

# Print ROUGE Scores
print('ROUGE Scores:')
print('Original Prompt Reformulation:', rouge_original_prompt_question_reformulate['rougeL'])
print('Few-Shot Prompt Reformulation:', rouge_few_shot_prompt_question_reformulate['rougeL'])
print('Instruction Prompt Reformulation:', rouge_instruction_prompt_question_reformulate['rougeL'])
print('Original Prompt Generated Response:', rouge_original_prompt_generated_response['rougeL'])
print('Few-Shot Prompt Generated Response:', rouge_few_shot_prompt_generated_response['rougeL'])
print('Instruction Prompt Generated Response:', rouge_instruction_prompt_generated_response['rougeL'], '\n')

# Print Semantic Similarity Scores (BERTScore)
print('Semantic Similarity (BERTScore) F1-Scores:')
print(f"Original Prompt Reformulation: {sum(bertscore_original_prompt_question_reformulate['f1']) / len(bertscore_original_prompt_question_reformulate['f1']):.4f}")
print(f"Few-Shot Prompt Reformulation: {sum(bertscore_few_shot_prompt_question_reformulate['f1']) / len(bertscore_few_shot_prompt_question_reformulate['f1']):.4f}")
print(f"Instruction Prompt Reformulation: {sum(bertscore_instruction_prompt_question_reformulate['f1']) / len(bertscore_instruction_prompt_question_reformulate['f1']):.4f}")
print(f"Original Prompt Generated Response: {sum(bertscore_original_prompt_generated_response['f1']) / len(bertscore_original_prompt_generated_response['f1']):.4f}")
print(f"Few-Shot Prompt Generated Response: {sum(bertscore_few_shot_prompt_generated_response['f1']) / len(bertscore_few_shot_prompt_generated_response['f1']):.4f}")
print(f"Instruction Prompt Generated Response: {sum(bertscore_instruction_prompt_generated_response['f1']) / len(bertscore_instruction_prompt_generated_response['f1']):.4f}")