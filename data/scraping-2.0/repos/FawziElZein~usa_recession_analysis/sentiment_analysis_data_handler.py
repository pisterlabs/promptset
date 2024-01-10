from lookups import Logger,ETLStep,FinvizWebScrape, DestinationDatabase, InputTypes, PoliticianSpeeches,ErrorHandling
from pandas_data_handler import return_data_as_df, return_insert_into_sql_statement_from_df, return_create_statement_from_df
from database_handler import execute_query
import os
from bs4 import BeautifulSoup
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import pandas as pd
from misc_handler import create_sql_table_index
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from logging_handler import show_logger_message, show_error_message

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")



def get_data_from_staging_table(db_session, columns, source_name, table_title, destination_schema_name):

    src_schema = destination_schema_name.value
    src_table = f"stg_{source_name.value}_{table_title.value}"

    select_query = "SELECT"

    if source_name.value == 'finviz':
        select_query += "\nCONCAT(ticker,'-',title) AS ticker_title,"

    for column in columns.value:
        select_query += '\n' + column + ','
    select_query = select_query[:-1]

    select_query += f"\n FROM {src_schema}.{src_table}"

    df = return_data_as_df(
        file_executor=select_query, input_type=InputTypes.SQL, db_session=db_session)
    df.set_index(df.columns[0], inplace=True)

    df = df[~df.index.duplicated(keep='first')]

    return df


def preprocess_text(text):
    # Preprocessing function
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    if isinstance(text, str):
        # r'[^\w\s]' : matches any character that is not a word character (alphanumeric or underscore) or a whitespace character
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        text = ' '.join(tokens)
    return text


def get_openAI_sentiment_result(text):
    sentiment_scores = []
    try:
        llm = OpenAI(
            openai_api_key=openai_api_key)
        request = f'This is a sentiment analysis request. Please send me the negative, neutral, positive and compound scores in a forme of array [negative,neutral,positive,compound]\n\n"{text}"\n\n'
        response = llm(request)

        start_index = response.find("[")
        end_index = response.find("]")
        sentiment_scores_str = response[start_index:end_index + 1]
        sentiment_scores = eval(sentiment_scores_str)
    except Exception as e:
        error_string_prefix = ErrorHandling.OPEN_AI_ERROR.value
        error_string_suffix = str(e)
        show_error_message(error_string_prefix,error_string_suffix)
    finally:
        return sentiment_scores


def analyze_sentiment(df, source_name, text_column):
    text_column = text_column.value
    df[['neg', 'neu', 'pos', 'compound']] = df[text_column].apply(get_openAI_sentiment_result).apply(
        lambda row: pd.Series(row, index=['neg', 'neu', 'pos', 'compound']))
    return df


def get_sentiment_analysis_results(db_session, resources):
    etl_step = ETLStep.HOOK.value
    logger_string_postfix = Logger.ANALYZE_SENTIMENTS.value
    show_logger_message(etl_step, logger_string_postfix)

    df_sentiment_list = []
    try:
        for resource in resources:

            df = get_data_from_staging_table(db_session, columns=resource.COLUMNS_NAME, source_name=resource.SOURCE,
                                            table_title=resource.TABLE_TITLE, destination_schema_name=DestinationDatabase.SCHEMA_NAME)

            if len(df):
                df_sentiment = analyze_sentiment(
                    df=df, source_name=resource.SOURCE, text_column=resource.TEXT_COLUMN_NAME)
                df_sentiment_list.append(
                    [resource.TABLE_TITLE.value, df_sentiment])
        


    except Exception as e:
        error_string_prefix = ErrorHandling.SENTIMENTS_RESULT_ERROR.value
        error_string_suffix = str(e)
        show_error_message(error_string_prefix,error_string_suffix)
    finally:
        return df_sentiment_list