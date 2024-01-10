import json
import os
import concurrent.futures
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from rake_nltk import Rake
from nltk.tokenize import sent_tokenize, word_tokenize
import time
import logging
import openai

# Configure logging
logging.basicConfig(level=logging.DEBUG)

rake = Rake()

# Load the OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    try:
        if isinstance(text, str):
            sentences = sent_tokenize(text)
            sentiment_scores = [sia.polarity_scores(sentence)["compound"] for sentence in sentences]
            return sum(sentiment_scores) / len(sentiment_scores) if sentences else 0
        else:
            return 0
    except Exception as e:
        logging.error(f"Error in getting sentiment: {str(e)}")
        return 0

def generate_chat_model(prompt, temperature, max_tokens=30):
    time.sleep(1)  # Add a delay between each API call
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Respond:"},
                      {"role": "user", "content": "Do not write lists."},
                      {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            presence_penalty=1
        )
        logging.debug(f"Generated chat model response: {response}")
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logging.error(f"Error in generating chat model response: {str(e)}")
        return ""

def matmol_responses(prompt, size):
    responses = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(size):
            layer_responses = []
            for j in range(size):
                # Adjust the temperature calculation
                temperature = (i / (size - 1) + j / (size - 1)) / 2 + 0.75
                num_responses = 2 if i == 0 or i == size - 1 or j == 0 or j == size - 1 else 3
                #num_responses = 3 if i == 0 or i == size - 1 or j == 0 or j == size - 1 else 4
                try:
                    node_responses = [(executor.submit(generate_chat_model, prompt, temperature).result(), (i, j)) for _ in range(num_responses)]
                    layer_responses.append(node_responses)
                    time.sleep(1)  # Add a delay between each API call
                except Exception as e:
                    logging.error(f"Error in matmol_responses: {str(e)}")
            responses.append(layer_responses)
    return responses

def get_keywords(text):
    try:
        if isinstance(text, str):
            rake.extract_keywords_from_text(text)
            ranked_phrases = rake.get_ranked_phrases()
            return ranked_phrases[0] if ranked_phrases else None
        else:
            return None
    except Exception as e:
        logging.error(f"Error in getting keywords: {str(e)}")
        return None

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def get_final_summary(responses, keywords):
    size = len(responses)
    half_size = max(size // 2, 1)

    summaries = [["" for _ in range(size)] for _ in range(size)]
    quadrant_keywords = [["" for _ in range(size)] for _ in range(size)]
    rejected = []

    max_retries = 2  # Maximum number of retries
    for i in range(size):
        for j in range(size):
            unique_found = False
            retries = 0
            while not unique_found and retries < max_retries:
                for k in range(len(responses[i][j])):
                    sentiment = get_sentiment(responses[i][j][k])

                    # Decide the quadrant based on the sentiment score
                    if sentiment >= 0.5:     # Highly positive
                        target_i, target_j = min(size - 1, i % half_size), min(size - 1, j % half_size)
                    elif sentiment >= 0:     # Mildly positive
                        target_i, target_j = min(size - 1, i % half_size), min(size - 1, j // half_size + half_size)
                    elif sentiment >= -0.5:  # Mildly negative
                        target_i, target_j = min(size - 1, i // half_size + half_size), min(size - 1, j % half_size)
                    else:                     # Highly negative
                        target_i, target_j = min(size - 1, i // half_size + half_size), min(size - 1, j // half_size + half_size)

                    # If the target cell is empty and the response is not too similar to existing summaries
                    if summaries[target_i][target_j] == "" and all(jaccard_similarity(word_tokenize(responses[i][j][k]), word_tokenize(summary)) < 0.5 for row in summaries for summary in row if summary):
                        summaries[target_i][target_j] = responses[i][j][k]
                        quadrant_keywords[target_i][target_j] = get_keywords(responses[i][j][k])
                        unique_found = True  # Stop retrying once a unique response is found
                    else:
                        rejected.append(responses[i][j][k])
                retries += 1

    final_summary = " ".join([f"{summaries[i][j]} ({quadrant_keywords[i][j]})" for i in range(size) for j in range(size) if summaries[i][j]]) 

    # Extract keywords from the final summary
    final_keywords = get_keywords(final_summary)
    overall_keywords = keywords + [final_keywords]  # Create a new list for overall keywords

    try:
        time.sleep(1)  # Add a delay between each API call
        # Generate the final summary using the chat model
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Summarise:"},
                      {"role": "user", "content": final_summary}],
            temperature=0.9,
            max_tokens=300
        )

        final_summary = response.choices[0].message['content'].strip()
        logging.debug(f"Generated final summary: {final_summary}")

    except Exception as e:
        logging.error(f"Error in get_final_summary: {str(e)}")

    return final_summary, overall_keywords  # Return the final summary and overall keywords as separate variables


def main():
    prompt = input("Enter your prompt: ")
    size = 2

    try:
        responses = matmol_responses(prompt, size)
        final_summary, overall_keywords = get_final_summary(responses, [get_keywords(prompt)])

        print("Summary:", final_summary)

        data = {
            'prompt': prompt,
            'final_summary': final_summary,
            'responses': [],
            'quadrant_summaries': []   # Add this to store the quadrant summaries
        }

        for i in range(size):
            for j in range(size):
                quadrant_responses = responses[i][j]
                quadrant_summary = generate_chat_model(" ".join([response for response, _ in quadrant_responses]), temperature=0.5, max_tokens=20)
                data['quadrant_summaries'].append({  # Add this to record the quadrant summary
                    'quadrant': f"{i+1}-{j+1}",
                    'summary': quadrant_summary
                })

                print(f"Quadrant {i+1}-{j+1} Summary:")
                print(quadrant_summary)
                print()

                for response, position in quadrant_responses:
                    response_data = {
                        'response': response,
                        'keywords': get_keywords(response),
                        'sentiment': get_sentiment(response),
                        'position': position
                    }
                    data['responses'].append(response_data)

                    # Print the generated response and its information
                    print("Response:", response)
                    print("Keywords:", response_data['keywords'])
                    print("Sentiment:", response_data['sentiment'])
                    print("Position:", response_data['position'])
                    print("-----------------------------")

        # Append overall keywords to the data dictionary
        data['overall_keywords'] = overall_keywords

        save_to_json(data, 'output.json')

    except Exception as e:
        print(f"Error in main: {str(e)}")



# Save the data to a JSON file
def save_to_json(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error in save_to_json: {str(e)}")

# Run the main function if this file is run directly
if __name__ == "__main__":
    main()
