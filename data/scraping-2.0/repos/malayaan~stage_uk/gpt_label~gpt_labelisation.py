import openai
import pandas as pd
from tqdm import tqdm
import re 
import multiprocessing

# Load the data
df = pd.read_csv(r'gpt_label\tweet.csv')

# Initialize OpenAI API
openai.api_key = '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'

def get_first_float(s):
    # Search for the first occurrence of a float in the string
    match = re.search(r'\d+(\.\d+)?', s)
    if match:
        # If a float is found, convert it to a float and return it
        return float(match.group())
    else:
        # If no float is found, return None
        return 2

# Function to get sarcasm/irony score
def get_sarcasm_irony_score(queue, tweet):
    try:
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Rate the following tweet: '{tweet}' for sartcasm and irony on a scale of 0 (no irony or sarcasm) to 1 (very ironic or sarcastic). Please provide only a float without any explanation."},
            ]
        )
        # Extract the score from the response
        response_content = response['choices'][0]['message']['content']
        score = get_first_float(response_content)
        queue.put((score, response_content))
    except Exception as e:
        queue.put(e)

def main():
    # Create or reset the 'sarcasm_irony_score' column if it doesn't exist
    if 'sarcasm_irony_score' not in df.columns:  
        df['sarcasm_irony_score'] = None

    # Create a tqdm instance
    pbar = tqdm(df.iterrows(), total=df.shape[0])

    # Apply the function to each tweet and save the scores
    for i, row in pbar:
        if pd.isnull(row['sarcasm_irony_score']):  # Only label tweets that haven't been labeled yet
            while True:
                queue = multiprocessing.Queue()
                p = multiprocessing.Process(target=get_sarcasm_irony_score, args=(queue, row['Tweet']))
                p.start()
                p.join(15)  # Allow the process to run for up to 15 seconds

                if p.is_alive():
                    print("Processing the tweet took longer than 15 seconds, retrying...")
                    p.terminate()
                    p.join()
                else:
                    result = queue.get()
                    if isinstance(result, Exception):
                        print(f"An error occurred: {result}. Retrying...")
                        continue
                    else:
                        score, response_content = result
                        df.loc[i, 'sarcasm_irony_score'] = score
                        pbar.set_postfix({"Last response": response_content[:50]})  # Display the first 50 characters of the response
                        # Save the dataframe to a new csv file
                        df.to_csv(r'gpt_label\tweet.csv', index=False)
                        break

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
