import duckdb
import openai
import pandas as pd
import time

from utils import *
from generate_summary import generate_summary


def get_data():

    conn = duckdb.connect('algolia_hn_search.duckdb')
    data = conn.sql('select * from hn_search_data.keyword_hits').df()
    conn.close()

    data['object_id'] = data['object_id'].map(int)
    data['parent_id'] = data['parent_id'].map(int)

    data = data.drop_duplicates(subset='object_id', keep='first')

    parent_data = data[["object_id","comment_text"]]
    parent_data.columns = ["parent_object_id","parent_comment"]

    merged_data = data.merge(
        parent_data,
        left_on="parent_id",
        right_on="parent_object_id",
        how="left"
    )

    merged_data['comment_text'] = merged_data['comment_text']\
                                    .fillna('').apply(clean_text)
    merged_data['story_title'] = merged_data['story_title']\
                                    .fillna('').apply(clean_text)
    merged_data['parent_comment'] = merged_data['parent_comment']\
                                    .fillna('').apply(clean_text)

    merged_data["relevance"] = merged_data.apply(
        lambda row: check_relevance(
            row["comment_text"], row["story_title"]
        ),
        axis=1
    )

    relevant_data = merged_data[merged_data["relevance"]]
    relevant_data['keywords'] = relevant_data.apply(
    	lambda row: extract_keywords(
    		row['comment_text'], row['story_title'], row['parent_comment']
    	),
    	axis=1
    )

    return relevant_data

if __name__ == '__main__':

    relevant_data = get_data()
    print(f'{len(relevant_data)} rows of relevant data found')

    rate_limit_counter = 1
    new_hits=True
    while new_hits:
        try:            
            try:
                input_data = pd.read_csv('dashboard_data.csv')
                existing_objects = input_data['object_id']
            except FileNotFoundError:
                input_data = pd.DataFrame()
                existing_objects = []
                
            new_data = relevant_data[
                ~relevant_data['object_id'].map(int).isin(existing_objects)
            ]
            print(f'{len(new_data)} new comments found in the relevant data')
            print(f'Extracting {min(len(new_data),30)} rows')

            new_data = new_data.head(30)

            if len(new_data):

                print('generating new comment summaries using gpt-3.5-turbo-0301 api ...')
                strt = time.time()
                new_data[["gpt_response","total_tokens"]] = new_data.apply(generate_summary, axis=1)
                print(f'new comment summaries generated in {round(time.time() - strt,2)}s')
                # new_data['keywords'] = new_data['gpt_response'].apply(extract_keywords)
                dashboard_data = pd.concat([input_data,new_data])
                print('updating existing data with the new data')
                dashboard_data.to_csv('dashboard_data.csv', index=False)

            else:
                new_hits=False

        except openai.error.RateLimitError:

            print(f'Encountered Rate Limit Error: Attempt {rate_limit_counter}')

            if rate_limit_counter > 10:
                print(f'Encountered Rate Limit Error 10 times. Aborting ...')
                new_hits=False
            else:
                rate_limit_counter += 1
                continue
