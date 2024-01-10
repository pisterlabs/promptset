import pandas as pd
import numpy as np
import re
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#%%
from .googlenewsapi import get_news
from .news_articles_paragraph import get_articles_content
#from tok_lem_pipeline import tokenize_whole_maude
#%%
from .roberta_classification import RobertaClassification
roberta_obj = RobertaClassification()
#%%

def analyze_news_articles(query, start_date, end_date, country, language, openai_api_key):
    news = get_news(query, start_date, end_date, country, language)
    news_df = pd.DataFrame(news)
    
    
    news_article_df = get_articles_content(news_df, para_length=1)
    #print('Total news articles: {}'.format(len(news_article_df)))
    #print(news_article_df.columns)
    
    return news_article_df

"""
def chat_completion(query, news_articles_df, openai_api_key):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
    )
    
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
      ],
      n=1,
      max_tokens=100,
      temperature=0.5,
    )
    
    generated_text = response.choices[0].message.content
    
    return generated_text

def similarity_check(generated_text, news_article_df, language):
    stop_words = {"the", "and", "of", "to", "in", "that", "is", "with", "for", "on"}
    
    words = re.findall(r'\w+', generated_text)

    # Create a set to store the unique keywords
    keywords = set()

    # Add each word to the set if it's not a stop word

    for word in words:
        if word.lower() not in stop_words:
            keywords.add(word.lower())
            

    # Convert the set of keywords to a list
    keywords = list(keywords)
    
    texts = keywords + news_article_df['text'].tolist()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute cosine similarity between each article and the keywords
    similarity_scores = cosine_similarity(tfidf_matrix[len(keywords):], tfidf_matrix[:len(keywords)])

    # Compute the mean similarity score for each article
    mean_scores = np.mean(similarity_scores, axis=1)

    # Set your threshold
    threshold = 0.0001  # replace with your threshold

    # Discard articles with a mean similarity score below the threshold
    news_article_df = news_article_df[mean_scores >= threshold]
    
    return news_article_df

"""
def tense_classification(news_article_df):
    categories_to_clasify = ['past', 'present', 'future']
    threshold = 0.70
    text_column = "text"

    # loop over dataframe and classify each row
    for index, row in news_article_df.iterrows():
        confidence_scores = roberta_obj.roberta_classifier(row[text_column], categories_to_clasify, threshold)
        if (confidence_scores['present'] + confidence_scores['future']) < threshold:
            news_article_df.drop(index, inplace=True)
    
    return news_article_df

def alert_classification(news_article_df):
    categories_to_clasify = ['Risk and Warning', 'Caution and Advice', 'Safe and Harmless']
    threshold = 0.50
    text_column = "text"


    # loop over dataframe and classify each row
    for index, row in news_article_df.iterrows():
        confidence_scores = roberta_obj.roberta_classifier(row[text_column], categories_to_clasify, threshold)

        # find the key with the highest value
        alert_class = max(confidence_scores, key=confidence_scores.get)
        alert_score = confidence_scores[alert_class]

        news_article_df.at[index, 'alert_class'] = alert_class
        news_article_df.at[index, 'alert_score'] = alert_score
        
        
    alert_dict = {}

    # get total length of news_df
    total_news = len(news_article_df)
    print('Total news: {}'.format(total_news))

    # total articles with alert_class = high alert
    high_alert = len(news_article_df[news_article_df['alert_class'] == 'Risk and Warning'])
    print('Risk and Warning: {}'.format(high_alert))

    # total articles with alert_class = low alert
    low_alert = len(news_article_df[news_article_df['alert_class'] == 'Caution and Advice'])
    print('Caution and Advice: {}'.format(low_alert))

    # total articles with alert_class = others
    others = len(news_article_df[news_article_df['alert_class'] == 'Safe and Harmless'])
    print('Safe and Harmless: {}'.format(others))

    # return the column text and alert_score for high alert articles with max and second max alert_score
    high_alert_df = news_article_df[news_article_df['alert_class'] == 'Risk and Warning']
    high_alert_df = high_alert_df.sort_values(by=['alert_score'], ascending=False)
    high_alert_df = high_alert_df[['text', 'alert_score']]
    high_alert_df = high_alert_df.reset_index(drop=True)

    # add to dictionary
    alert_dict['Total news'] = total_news
    alert_dict['Risk and Warning'] = high_alert
    alert_dict['Caution and Advice'] = low_alert
    alert_dict['Others'] = others

    if len(high_alert_df) == 0:
        alert_dict['First high alert text'] = None
        alert_dict['First high alert score'] = None
        alert_dict['Second high alert text'] = None
        alert_dict['Second high alert score'] = None
    elif len(high_alert_df) == 1:
        alert_dict['First high alert text'] = high_alert_df['text'][0]
        alert_dict['First high alert score'] = high_alert_df['alert_score'][0]
        alert_dict['Second high alert text'] = None
        alert_dict['Second high alert score'] = None
    else: # len(high_alert_df) > 1
        alert_dict['First high alert text'] = high_alert_df['text'][0]
        alert_dict['First high alert score'] = high_alert_df['alert_score'][0]
        alert_dict['Second high alert text'] = high_alert_df['text'][1]
        alert_dict['Second high alert score'] = high_alert_df['alert_score'][1]
        
    return alert_dict
#%%

def full_pipeline(query, start_date, end_date, country, language):
    # Step 1: Get News Articles
    news_articles_df = analyze_news_articles(query, start_date, end_date, country, language, openai_api_key)

    # Step 2: Chat Completion
    #generated_text = chat_completion(query, news_articles_df, openai_api_key)

    # Step 3: Similarity Check
    #cosined_articles_df = similarity_check(generated_text, news_articles_df, language)

    # Step 4: Tense Classification
    future_articles_df = tense_classification(news_articles_df)

    # Step 5: Alert Classification
    alert_dict = alert_classification(future_articles_df)

    return alert_dict