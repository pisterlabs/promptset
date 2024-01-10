import os
import pandas as pd
from modules import setup_logger
from modules import OpenAIRequest


# Initialize logger.
logger = setup_logger(__name__, "logs/extract_features.log")


'''
This program extracts features from tweets using the OpenAI API and stores them
in a csv file. The features are extracted using the GPT-3.5-turbo language model.

The program is designed to extract features in such a way that it can be stopped
and restarted at any time. The program will load the tweets from the `tweets.csv`
file and the features from the `tweets_gpt_features.csv` file. It will then
compare the two files and extract the features from the tweets that are not
already in the `tweets_gpt_features.csv` file.
'''


class FeatureExtraction:
    def __init__(self, df_path: str, results_df_path: str) -> None:

        # Initialize parameters.
        self.df_path = df_path
        self.results_df_path = results_df_path

    def extract_features(self):
        df = pd.read_csv(self.df_path)
        df = df.drop_duplicates(subset=['tw_texto'], keep='first')
        logger.info('`tweets.csv` has been loaded')

        try:
            df_results = pd.read_csv(self.results_df_path)
            df_results = df_results.drop_duplicates(subset=['tw_texto'], keep='first')
            logger.info(f'`tweets_gpt_features.csv` has been loaded, there are {len(df) - len(df_results)} rows to process')
        except FileNotFoundError:
            df_results = pd.DataFrame()
            logger.info('`tweets_gpt_features.csv` has been finitialized')

        df_to_process = df[~df['tw_texto'].isin(df_results['tw_texto'])]
        df_to_process = df_to_process.dropna()

        logger.info('Processing rows for GPT zero-shot feature extraction')
        for index, row in df_to_process.iterrows():
            tweet = row['tw_texto']
            candidate = row['candidato']

            logger.info(f"Starting API request for tweet: {tweet}")
            response = (
                OpenAIRequest(tweet)
                .preprocess_text()
                .extract_features(prefix='tw_')
            )
            logger.info(f"GPT response: {response}")

            df_result = pd.DataFrame([response], index=[index])

            # Add the tweet and candidate to the DataFrame.
            df_result['tw_texto'] = tweet
            df_result['tw_candidate'] = candidate

            # Reorder the columns.
            cols = df_result.columns.tolist()
            cols = ['tw_texto', 'tw_candidate'] + [col for col in cols if col not in ['tw_texto', 'tw_candidate']]
            df_result = df_result[cols]

            # Append to the results DataFrame.
            df_results = pd.concat([df_results, df_result])

            # Save to file after each request in case of failure.
            df_results.to_csv(self.results_df_path, index=False)


def main() -> None:
    df_path = os.path.join(os.getcwd(), 'data', 'tweets.csv')
    results_df_path = os.path.join(os.getcwd(), 'data', 'tweets_gpt_features.csv')
    batch = FeatureExtraction(df_path, results_df_path)
    batch.extract_features()


if __name__ == "__main__":
    main()