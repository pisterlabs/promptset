from geopy.geocoders import Nominatim
import pandas as pd
import re
from unidecode import unidecode
from datetime import datetime
import openai
import sqlalchemy
import time
import geopy
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

openai.api_key = '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'

app = Nominatim(user_agent="tutorial")

engine = sqlalchemy.create_engine(
    'mysql+pymysql://admin:N6zmVKVW@jobs-intelligence-slovakia.'
    'cluster-c0rbbiliflyo.eu-central-1.rds.amazonaws.com:9906/General_Intelligence_CZ')

conn = engine.connect()

def get_companies_slovakia():
    try:
        query = sqlalchemy.text('SELECT * FROM `Companies_Slovakia_Processed`')

        # Read data from the query
        dataframe_companies = pd.read_sql_query(query, conn)
        return dataframe_companies
    except Exception as e:
        print(e)

def get_CZ_WebCrawlResults():
    try:
        query = sqlalchemy.text('SELECT * FROM `CZ_WebCrawlResults`')

        # Read data from the query
        dataframe = pd.read_sql_query(query, conn)
        return dataframe
    except Exception as e:
        print(e)

def remove_numbers(string):
    if type(string) == str:
        return re.sub(r'\d+', '', string)


def split_word_by_comma(word):
    if word is not None and 'Praca vyzaduje cestovanie' in word:
        return 'Traveling job'
    elif word is not None:
        return [x.strip() for x in word.split(',') and word.split('-') and word.split(', ')]

def get_Cities_Processed():
    try:
        query = sqlalchemy.text('SELECT * FROM `Cities_Processed`')

        # Read data from the query
        dataframe = pd.read_sql_query(query, conn)
        return dataframe
    except Exception as e:
        print(e)

#function to drop table
def drop_table(table_name):
    try:
        query = sqlalchemy.text(f'DROP TABLE {table_name}')
        conn.execute(query)
    except Exception as e:
        print(e)

def get_zipcode(df, geolocator, lat_field, lon_field):
    #df.apply(get_zipcode, axis=1, geolocator=geolocator, lat_field='Lat', lon_field='Lon')
    location = geolocator.reverse((df[lat_field], df[lon_field]))
    return location.raw['address']['postcode']


def clean_locations(specific_location):
    # read database
    database_webcrawl_results = get_CZ_WebCrawlResults()
    database_cities = get_Cities_Processed()

    # place = 'Bratislava II, Bratislava, Slovakia (Job with occasional home office)'
    # words = split_word_by_comma(place)

    cities_list = database_cities[specific_location].to_list()

    # cities_list = cities_list[:10]
    # convert all characters in cities_list to ascii
    cities_list = [unidecode(x) for x in cities_list]

    # remove '-' from database_webcrawl_results['location'] only if it is not None
    database_webcrawl_results['location'] = database_webcrawl_results['location'].apply(
        lambda x: x.replace('-', '') if x is not None else x)
    database_webcrawl_results['location'] = database_webcrawl_results['location'].apply(
        lambda x: x.replace(' ,', ',') if x is not None else x)

    # loop through webcrawl results
    for index, row, in database_webcrawl_results.iterrows():
        # delete numbers from row['location']
        database_webcrawl_results.loc[index, 'location'] = remove_numbers(row['location'])
        # remove space from row['location']
        # database_webcrawl_results.loc[index, 'location'] = delete_space(row['location'])

        words = split_word_by_comma(row['location'])
        # loop through words and if one of the words is in database_cities['City'] then set the location to that city
        if words is not None:
            for word in words:
                if word in cities_list:
                    # database_webcrawl_results.loc[index, 'location'] = word
                    database_webcrawl_results.loc[index, specific_location] = word
                    break

    # drop webcrawl results table
    drop_table('CZ_WebCrawlResults')

    # import to mysql SK_WebCrawlResults table
    database_webcrawl_results.to_sql(name='CZ_WebCrawlResults', con=engine, if_exists='append', index=False)

#fill missing locations
def fill_locations():
    # read database
    database_webcrawl_results = get_CZ_WebCrawlResults()
    database_cities = get_Cities_Processed()

    # merge on Municipality only if City is None
    mask1 = database_webcrawl_results['Okres'].isnull()
    merged_df1 = pd.merge(database_webcrawl_results[mask1], database_cities, on='Obec', how='left')
    # delete columns City_x and Region_x
    merged_df1.drop(['Okres_x', 'Kraj_x'], axis=1, inplace=True)
    # rename columns City_y and Region_y to City and Region
    merged_df1.rename(columns={'Okres_y': 'Okres', 'Kraj_y': 'Kraj'}, inplace=True)
    #drop rows where Municipality is None
    merged_df1 = merged_df1[merged_df1['Obec'].notna()]


    mask = database_webcrawl_results['Obec'].isnull()
    merged_df2 = pd.merge(database_webcrawl_results[mask], database_cities, on='Okres', how='left')
    #delete columns Municipality_x and Region_x
    merged_df2.drop(['Obec_x', 'Kraj_x'], axis=1, inplace=True)
    #delete rows where City is None
    #merged_df2 = merged_df2[merged_df2['City'].notna()]
    #rename columns Municipality_y and Region_y to Municipality and Region
    merged_df2.rename(columns={'Obec_y': 'Obec', 'Kraj_y': 'Kraj'}, inplace=True)

    #if region in database_webcrawl_results is filled but Municipality is None and City is None then fill Municipality and City with string unspecified
    database_webcrawl_results.loc[(database_webcrawl_results['Kraj'].notna()) & (database_webcrawl_results['Obec'].isnull()) & (database_webcrawl_results['Okres'].isnull()), ['Okres', 'Obec']] = 'unspecified'
    #keep only rows where Municipality and City are unspecified
    database_webcrawl_results = database_webcrawl_results[(database_webcrawl_results['Okres'] == 'unspecified') & (database_webcrawl_results['Obec'] == 'unspecified')]

    #concatenate merged_df1 and merged_df2 with database_webcrawl_results
    database_webcrawl_results_merged = pd.concat([merged_df1, merged_df2, database_webcrawl_results], ignore_index=True)
    # fill missing values in latitude, longitude and Zipcode with unspecified
    database_webcrawl_results_merged['Latitude'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['Longitude'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['Kód okresu'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['Kód kraje'].fillna('unspecified', inplace=True)

    #database_webcrawl_results_merged['work_type'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['Obec'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['Okres'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['Kraj'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['salary'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['PSČ'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['Kód obce'].fillna('unspecified', inplace=True)

    #copy index to column id
    database_webcrawl_results_merged['index'] = database_webcrawl_results_merged.index

    #drop database webcrawl_results_table
    drop_table('CZ_WebCrawlResults')

    #import to mysql SK_WebCrawlResults table
    database_webcrawl_results_merged.to_sql(name='CZ_WebCrawlResults', con=engine, if_exists='append', index=False)

def fill_locations_companies():
    # read database
    database_companies = get_companies_slovakia()
    database_cities = get_Cities_Processed()

    # merge on City
    merged_df = pd.merge(database_companies, database_cities, on='City', how='left')

    # drop table
    drop_table('Companies_Slovakia_Processed')

    # import to mysql Companies_Slovakia_Processed table
    merged_df.to_sql(name='Companies_Slovakia_Processed', con=engine, if_exists='append', index=False)

def location_cleaning():
    for location in ['Obec', 'Okres', 'Kraj']:
        clean_locations(location)

if __name__ == '__main__':
    #location_cleaning()
    fill_locations()

