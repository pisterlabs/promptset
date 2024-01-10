import concurrent
import json
import math
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import openai
import time

def process_query(query):
    # Your processing logic for the query goes here
    # You need to implement the necessary steps to process the query and return a digit as the score.
    # Replace the following line with your own implementation
    api = "Your API key"
    # Extract Conversion

    sentence2 = "Dave (1993). Reference: Movie Name: Glengarry Glen Ross (1992), Rating: 4;Movie Name: Fargo (1996), Rating: 4;Movie Name: Babe (1995), Rating: 4;Movie Name: Seventh Seal, The (Sjunde inseglet, Det) (1957), Rating: 3;Movie Name: Grease (1978), Rating: 2;"
    response2 = "Based on the reference movies and their ratings, it seems like the user enjoys a wide variety of films, but seems to have a preference for higher-rated films. Since \"Dave\" is a comedy-drama, similar in genre to some of the higher-rated movies in the list, the possible score this user would give to \"Dave\" might be a \"4\"."
    sentence3 = "Please answer reasoning first. And output the rating as an int at the end of your response. Please also consider the cast, country, lanuage, genres and year."
    response3 = "Let's break it down:\n\n\"Glengarry Glen Ross\" is a drama with intense performances, which the user rated highly (4).\n\"Fargo\" is a mix of dark comedy and crime, again rated highly (4) by the user.\n\"Babe\" is a comedy-drama film, with a lighter, more family-oriented theme, still rated high (4).\n\"The Seventh Seal\" is a classic existential drama, more serious in tone, rated a bit lower (3).\n\"Grease\" is a musical, which is a different genre altogether, and the user rated it as the lowest (2).\nGiven this, it's clear that the user leans towards dramas and dark comedies, with a preference for intense performances or unique storytelling.\n\n\"Dave\" is a comedy-drama, lighter in tone but with a unique premise, and might appeal to the user's taste, as seen with the high rating for \"Babe\". However, considering that it isn't as intense as \"Glengarry Glen Ross\" or \"Fargo\", it might not reach the highest rating.\n\n4"


    # query = query[0].replace(";",",")
    # print(query)
    # print(query)
    # exit()
    # try:    time.sleep(4)
    # time.sleep(4)
    # query = query
    a = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            # {"role": "user", "content": sentence1},
            # {"role": "assistant", "content": response1},
            {"role": "user", "content": sentence2},
            {"role": "assistant", "content": response2},
            {"role": "user", "content": sentence3},
            {"role": 'assistant', "content": response3},
            {"role": 'user', "content":query}

        ],
        api_key=api,
        n=3,

    )
    # print(a["choices"][0]["message"]["content"])


    for i in range(3):
        try:
            print(query)
            print("\n")
            print(a["choices"][i]["message"]["content"])
            # print(a["choices"][i]["message"]["content"][-3])
            score = a["choices"][i]["message"]["content"][-3]
            score = int(score)
            break
        except:
            score = 3
            continue

    # except:
    #     print('wrong!')
    #     score  = 3
    return score


def evaluate_query(item):
    query = item['query']
    # print(query)
    # exit()
    real_score = item['rating']

    # Process the query and get the predicted score
    predicted_score = process_query(query)

    return predicted_score, real_score


def evaluate_predictions(data, checkpoint_file=None, num_threads=4):
    predictions = []
    real_scores = []
    checkpoint = 0

    if checkpoint_file:
        try:
            with open(checkpoint_file) as file:
                checkpoint_data = json.load(file)
                checkpoint = checkpoint_data['checkpoint']
                predictions = checkpoint_data['predictions']
                real_scores = checkpoint_data['real_scores']
        except FileNotFoundError:
            pass

    with tqdm(total=len(data[checkpoint:])) as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_item = {executor.submit(evaluate_query, item): item for item in data[checkpoint:]}

            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    predicted_score, real_score = future.result()

                    # Append the predicted score and real score to the respective lists
                    predictions.append(predicted_score)
                    real_scores.append(real_score)
                except Exception as e:
                    print(f"Error processing query: {item['query']}. Error: {e}")

                # Update progress bar
                pbar.update(1)

                # Save intermediate results to checkpoint file
                if checkpoint_file:
                    checkpoint_data = {
                        'checkpoint': checkpoint + 1,
                        'predictions': predictions,
                        'real_scores': real_scores
                    }
                    with open(checkpoint_file, 'w') as file:
                        json.dump(checkpoint_data, file)

    # Compute RMSE
    rmse = math.sqrt(sum((p - r) ** 2 for p, r in zip(predictions, real_scores)) / len(predictions))

    # Compute Accuracy
    accuracy = accuracy_score(real_scores, predictions)

    # Compute Recall
    recall = recall_score(real_scores, predictions, average='weighted')

    # Compute Precision
    precision = precision_score(real_scores, predictions, average='weighted')

    # Compute F1 Score
    f1 = f1_score(real_scores, predictions, average='weighted')

    return rmse, accuracy, recall, precision, f1


# Read data from 'LLM_eval.json' file
def read_data_from_file(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data


# Example usage with your LLM_eval.json data
file_path = 'LLM_eval.json'
data = read_data_from_file(file_path)
temp_data = []
keys = list(data.keys())[:500]
for key in keys:
    temp_data.append(data[key])
data = temp_data

checkpoint_file = 'evaluation_checkpoint.json'  # Specify the checkpoint file path
num_threads = 10  # Number of threads for concurrent processing

rmse, accuracy, recall, precision, f1 = evaluate_predictions(data, checkpoint_file,num_threads=num_threads)
print("RMSE:", rmse)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)
