from sklearn.metrics import confusion_matrix
import openai,re,pickle,os,textwrap
import pandas as pd
import numpy as np
from tqdm import tqdm

def cleanup(text):
    text=re.sub(r"[^a-z0-9\.\?\! ]", " ",text.lower())
    text=re.sub(r'\.+','.',text)
    text=' '.join(text.split()).strip()
    return text
# Load the dataset
data = pd.read_csv('sorted_data_acl.csv', keep_default_na=False,delimiter=',')

labels=data['sentiment'].apply(lambda x: 0 if x[0] == 'p' else 1)
# Convert rating to human language sentence that lang transformer can pay attention to
data['rating']=data['rating'].apply(lambda x:  'user rating '+str(int(x)))
# Convert usefull to human language sentence that lang transformer can pay attention to
data['helpful']=data['helpful'].apply(lambda x:  'deemed helpful by '+x if len(x) else x)
# Merge to one one review text but keep just lowercase alnum and cleanup \n
# keep !? those carry strong emotions too
# example train row text: user rating 1. deemed helpful by 4 of 9. horrible book horrible. the..
texts=(data['rating']+'. '+data['helpful']+'. '+data['title']).apply(cleanup)+'; '+data['review_text'].apply(cleanup)
results=[]

# Initialize variables to keep track of predictions and actual labels
predictions = []
actual_labels = []

# Initialize confusion matrix
conf_matrix = np.zeros((2, 2))

# Run sentiment analysis and fill in confusion matrix
def GPT4_get_chunk_sentiment(entry):
    global conf_matrix,actual_labels,predictions
    prompt=f"You are a sentiment classifier. Focus on the rating number, how many people found the review useful, and then the review text itself. Review: {entry}. What is the sentiment? positive or negative? and how sure you are between 0-1"
    messages=[{"role": "system", "content": prompt}]
    completion = openai.ChatCompletion.create(messages=messages,model="gpt-4",  max_tokens=15000)

    pred_label = 1 if 'positive' in completion.choices[0].message.content.strip() else 0
    predictions.append(pred_label)
    
    # Update the confusion matrix
    conf_matrix[labels, pred_label] += 1
    mt=['positive','negative']
    return pred_label

results=[]
if os.path.exists('results3.pickle'):
    with open('results3.pickle','rb') as f:
        results= pickle.load(f)
else:
    for text in tqdm(texts[:100]):

        short_summary_and_numbers,long_review_text=text.split(';')
        paragraphs = textwrap.wrap(long_review_text, 5000*2.5)

        subresults=[]
        for paragraph in paragraphs:
            res=0
            try:
                res=GPT4_get_chunk_sentiment(short_summary_and_numbers.strip()+' '+paragraph.strip())
            except:
                pass
            subresults.append(res)

        # Calculate the average sentiment
        avg_sentiment = round(sum(subresults) / len(subresults))
        
        results.append(avg_sentiment)

        with open('results3.pickle','wb') as f:
            pickle.dump(results,f)

from sklearn.metrics import confusion_matrix
results = [1 - x for x in results]
labels=labels[:100]
# Calculate the confusion matrix
cm = confusion_matrix(labels, results)

# Calculate the accuracy
accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

# Print the results
print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(cm)