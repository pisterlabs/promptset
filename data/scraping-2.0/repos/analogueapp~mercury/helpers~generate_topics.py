import re
import os
import openai
from models.utils import embedding_predictor

openai.api_key = os.getenv("OPENAI_KEY")

def cls_pooling(model_output):
    # first element of model_output contains all token embeddings
    return [sublist[0] for sublist in model_output][0]

def get_embedding(topic):
    data = {
        "inputs": topic
    }

    res = cls_pooling(embedding_predictor.predict(data=data))    
    return res

def generate_topics(medium, title, specifier, num = 'a few'):    
    prompt_text = (f"List {num} relevant genres or topics for the {medium} '{title}' {specifier}. "
                   f"These genres or topics should be concise and reproducible across other works, "
                   f"representing a balanced mix of specificty to facilitate effective recommendations in a recommender engine. "                   
                   f"Simply provide a list of the genres or topics; do not include the title of the work itself or an elaboration of the topic.")

    # Make the API call
    completion = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=prompt_text, max_tokens=3000, temperature=0.6, n=1)
                
    # Extracting the response and splitting into individual topics    
    raw_text = completion.choices[0].text.strip()    
    topics = re.findall(r'\d+\.\s*(.*?)(?=\n\d+|$)', raw_text)    
            
    # Strip trailing spaces around each parsed topic
    topics = [topic.strip() for topic in topics]

    # print(topics)
    return topics

# if __name__ == "__main__":
#     generate_topics("film", "Blade Runner", "from 1982", 5)