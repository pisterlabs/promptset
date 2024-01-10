import os
from geopy.geocoders import Nominatim
import pandas as pd
import re
from unidecode import unidecode
import openai
import sqlalchemy


openai.api_key = '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'

app = Nominatim(user_agent="tutorial")

engine = sqlalchemy.create_engine(
    'mysql+pymysql://admin:N6zmVKVW@jobs-intelligence-slovakia.'
    'cluster-c0rbbiliflyo.eu-central-1.rds.amazonaws.com:9906/General_Intelligence')

conn = engine.connect()


def remove_numbers(string):
    if type(string) == str:
        return re.sub(r'\d+', '', string)

def get_kraj(address):
    if address is not None:
        try:
            location = get_location_by_address(fr'Slovensko, {address}')['display_name']
            return extract_kraj(location)
        except Exception as e:
            return None
    else:
        return None

def delete_space(string):
    #delete \n from string
    if type(string) == str:
        return string.replace(", ", "")

def extract_kraj(s):
    match = re.search(r'(\w+\skraj)', s)
    if match:
        return match.group(1)
    else:
        return None

def get_location_by_address(address):
    """This function returns a location as raw from an address
    will repeat until success"""
    time.sleep(1)
    try:
        return app.geocode(address).raw
    except:
        return {'lat': None, 'lon': None}

#get longitude and latitude
def get_longitude(address):
    """This function returns a tuple of (latitude, longitude) from an address"""
    location = get_location_by_address(address)
    if location['lon'] is None:
        return None
    return location['lon']

#get longitude and latitude
def get_latitude(address):
    """This function returns a tuple of (latitude, longitude) from an address"""
    location = get_location_by_address(address)
    if location['lat'] is None:
        return None
    return location['lat']

#function that finds string inside another string and replaces it with another string
def find_and_replace(string, substrings):
    if type(string) == str:
        try:
            string = unidecode(string)
        except Exception as e:
            pass
        for substring in substrings:
            if string.find(substring) != -1:
                return substring
            elif string == 'NaN':
                return 'NaN'
    return string

def remove_string_from_list(string, string_list):
    unidecode(string)
    for item in string_list:
        unidecode(item)
        if item in string:
            return None
        else:
            return string

def get_companies_slovakia():
    try:
        query = sqlalchemy.text('SELECT * FROM `Companies_Slovakia_Processed`')

        # Read data from the query
        dataframe_companies = pd.read_sql_query(query, conn)
        return dataframe_companies
    except Exception as e:
        print(e)

def get_sk_WebCrawlResults():
    try:
        query = sqlalchemy.text('SELECT * FROM `SK_WebCrawlResults`')

        # Read data from the query
        dataframe = pd.read_sql_query(query, conn)
        return dataframe
    except Exception as e:
        print(e)

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


def split_word_by_comma(word):
    if word is not None and 'Praca vyzaduje cestovanie' in word:
        return 'Traveling job'
    elif word is not None:
        return [x.strip() for x in word.split(',') and word.split('-') and word.split(', ')]


def get_zipcode(df, geolocator, lat_field, lon_field):
    #df.apply(get_zipcode, axis=1, geolocator=geolocator, lat_field='Lat', lon_field='Lon')
    location = geolocator.reverse((df[lat_field], df[lon_field]))
    return location.raw['address']['postcode']

def get_pkl_with_latest_date():
    #read all pickle files in directory
    import glob
    files = glob.glob(r'C:\Users\labus.INTERCONNECTION\Desktop\WebScrapingProject\Data\JobsIntelligence\Slovak\Unprocessed\*.pkl')

    #get the latest date from the files
    latest_date = max(files, key=os.path.getctime)

    #return the file_name
    return latest_date

def clean_locations(database_webcrawl_results, specific_location):
    # read database
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

    return database_webcrawl_results

#fill missing locations
def fill_locations(database_webcrawl_results):
    # read database
    database_cities = get_Cities_Processed()

    # if column Region is not in database_webcrawl_results then create it and fill it with None
    if 'Region' not in database_webcrawl_results.columns:
        database_webcrawl_results['Region'] = None



    # merge on Municipality only if City is None
    mask1 = database_webcrawl_results['City'].isnull()
    merged_df1 = pd.merge(database_webcrawl_results[mask1], database_cities, on='Municipality', how='left')
    # delete columns City_x and Region_x
    merged_df1.drop(['City_x', 'Region_x'], axis=1, inplace=True)
    # rename columns City_y and Region_y to City and Region
    merged_df1.rename(columns={'City_y': 'City', 'Region_y': 'Region'}, inplace=True)
    #drop rows where Municipality is None
    merged_df1 = merged_df1[merged_df1['Municipality'].notna()]

    mask = database_webcrawl_results['Municipality'].isnull()
    merged_df2 = pd.merge(database_webcrawl_results[mask], database_cities, on='City', how='left')
    #delete columns Municipality_x and Region_x
    merged_df2.drop(['Municipality_x', 'Region_x'], axis=1, inplace=True)
    #delete rows where City is None
    #merged_df2 = merged_df2[merged_df2['City'].notna()]
    #rename columns Municipality_y and Region_y to Municipality and Region
    merged_df2.rename(columns={'Municipality_y': 'Municipality', 'Region_y': 'Region'}, inplace=True)

    #if region in database_webcrawl_results is filled but Municipality is None and City is None then fill Municipality and City with string unspecified
    database_webcrawl_results.loc[(database_webcrawl_results['Region'].notna()) & (database_webcrawl_results['Municipality'].isnull()) & (database_webcrawl_results['City'].isnull()), ['Municipality', 'City']] = 'unspecified'
    #keep only rows where Municipality and City are unspecified
    database_webcrawl_results = database_webcrawl_results[(database_webcrawl_results['Municipality'] == 'unspecified') & (database_webcrawl_results['City'] == 'unspecified')]

    #concatenate merged_df1 and merged_df2 with database_webcrawl_results
    database_webcrawl_results_merged = pd.concat([merged_df1, merged_df2, database_webcrawl_results], ignore_index=True)
    # fill missing values in latitude, longitude and Zipcode with unspecified
    database_webcrawl_results_merged['latitude'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['longitude'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['ZIP_CODE'].fillna('unspecified', inplace=True)
    #database_webcrawl_results_merged['work_type'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['Municipality'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['City'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['Region'].fillna('unspecified', inplace=True)
    database_webcrawl_results_merged['salary'].fillna('unspecified', inplace=True)

    return database_webcrawl_results_merged


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

def location_cleaning(dataframe_location):
    for location in ['City', 'Municipality', 'Region']:
        dataframe_location = clean_locations(dataframe_location,location)
    return dataframe_location

def process_locations(dataframe):
    dataframe = location_cleaning(dataframe)
    return fill_locations(dataframe)


#main
if __name__ == '__main__':
    process_locations()