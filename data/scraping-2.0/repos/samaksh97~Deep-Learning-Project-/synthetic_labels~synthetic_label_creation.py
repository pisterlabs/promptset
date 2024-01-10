from openai import OpenAI
import time
import pandas as pd
from dotenv import dotenv_values

config = dotenv_values('../.env.local')

# Gets the key from the OPENAI_API_KEY environment variable in .env file
client = OpenAI(
  api_key = config['OPENAI_API_KEY'],
)  

# Define a function to process a batch of reviews
def process_reviews_batch(reviews_batch):
    # Initialize an empty list to store API responses
    batch_responses = []

    messages = messages=[
                {"role": "system", "content": "You are an assistant which can classify each paper in the list based on its title to one of the five following conferences: 'VLDB', 'ISCAS', 'SIGGRAPH', 'INFOCOM', 'WWW' and return answers as a numbered list."}]
    
    batch_str = ''
    # Process each review in the batch
    for i, review in enumerate(reviews_batch):
        batch_str += f'{i+1}. {review}\n'
    messages.append({"role": "user", "content": batch_str})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    res = response.choices[0].message.content.split("\n")
    temp = []
    labels = []
    for item in res:
        try:
            splt = item.split('. ')
            temp.append(splt[0])
            labels.append(splt[1].replace("'", ''))
        except:
            print(f'Error in processing the response for {item}')
        
    res = [(reviews_batch[int(i)-1], labels[temp.index(i)]) for i in temp]
    batch_responses.extend(res)

    return batch_responses


if __name__ == '__main__':
    # Read the dataset
    df = pd.read_csv('../data/title_conference.csv')
    
    
    reviews = df['Title'].tolist()
    # reviews = reviews[240:330]
    # sub_df = pd.DataFrame({'Title': reviews})

    # Define the batch size
    batch_size = 30
    
    # Initialize an empty list to store the results
    results = []
    
    # Process the reviews in batches
    for i in range(0, len(reviews), batch_size):
            
        # Get a batch of reviews
        reviews_batch = reviews[i:i + batch_size]

        # Process the batch of reviews
        batch_results = process_reviews_batch(reviews_batch)
        
        print(f'Finished for Batch --> {i} to {i+batch_size}')
        # Store the results
        results.extend(batch_results)
        
        if i%5 == 0:
            with open('../data/synthetic_labels.txt', 'w') as f:
                f.writelines(str(results))
                # f.write('\n'.join(str(results)))
                
        # Add a small delay to avoid API rate limits
        time.sleep(2)

    # Print the results for each review
    try:
        final_df = pd.DataFrame(results, columns=['Title', 'Synthetic_Labels'])
        df.merge(final_df, on='Title', how='left').to_csv('../data/synthetic_labels.csv', index=False)
    except:
        print('Error in saving the synthetic labels')