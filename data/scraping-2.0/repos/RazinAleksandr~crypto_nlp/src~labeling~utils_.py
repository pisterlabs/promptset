import openai
import numpy as np


class SpeakEasy:
    def __init__(self, key, message_history, model_id, limit=5):
        key = key
        self.message_history = message_history
        self.model_id = model_id
        self.limit = limit
        
        openai.api_key = open(key, "r").read().strip("\n")
        

    def predict(self, question):
        # tokenize the new question sentence
        self.message_history.append({"role": "user", "content": f"{question}"})

        completion = openai.ChatCompletion.create(
            model=self.model_id,  # 10x cheaper than davinci, and better. $0.002 per 1k tokens
            messages=self.message_history
        )
        # Just the reply:
        reply_content = completion.choices[0].message.content#.replace('```python', '<pre>').replace('```', '</pre>')

        #print(reply_content)
        self.message_history.append({"role": "assistant", "content": f"{reply_content}"})
        
        history = {
            'request': {
                'question': [],
                'response': []
            }
        }
        history['request']['question'].append(
                    self.message_history[-2]['content']
                )
        history['request']['response'].append(
                    self.message_history[-1]['content']
                )
        if len(self.message_history) > self.limit + 2:
            self.message_history = self.message_history[:2] + self.message_history[-4:]

        return history
    

    def run_answer(self, title):
        history = self.predict(title)
    
        prediction = {'title': title,
                'sentiment': None}
        
        for i, (k, v) in enumerate(history['request'].items()):
            if i == 0: continue
            prediction['sentiment'] = v
        
        return prediction
    

def batch_generator(df, text_column, batch_size):
    num_batches = int(np.ceil(len(df) / batch_size))
    while True:
        #df = df.sample(frac=1)  # shuffle the dataframe
        for i in range(num_batches):
            batch = df[i*batch_size : (i+1)*batch_size]
            X_batch = batch[text_column].values  # replace 'text_column' with the name of your text column
            #y_batch = batch['label_column'].values  # replace 'label_column' with the name of your label column
            yield X_batch#, y_batch


def extract_classes(s):
    # Remove any leading/trailing whitespace and punctuation
    s = s.strip().rstrip('.').rstrip(',').lower()
    # Split the string into a list of individual class strings
    class_list = s.split(', ')
    # Map each class string to either 'positive', 'negative', or 'neutral'
    class_map = {'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'}
    class_list = [class_map.get(c, 'unknown') for c in class_list]
    # Remove any 'unknown' classes that were not mapped
    class_list = [c for c in class_list if c != 'unknown']
    return class_list
