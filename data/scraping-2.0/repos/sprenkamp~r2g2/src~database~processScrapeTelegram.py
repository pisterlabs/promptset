import os
from dotenv import load_dotenv
load_dotenv()
import argparse
from pymongo import MongoClient
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
import pandas as pd
import datetime
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

def validate_local_file(f):  # function to check if file exists
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def get_chats_list(input_file_path):
    """
    Args:
        input_file_path: chats path

    Returns: pandas dataframe. e.g.
            |country|chat|
            |Switzerland|https://t.me/zurich_hb_help|
            |Switzerland|https://t.me/helpfulinfoforua|
    """
    countries, chats = list(), list()
    with open(input_file_path, 'r') as file:
        for line in file.readlines():
            if line.startswith("#"):
                country = line.replace('#', '').replace('\n', '')
            else:
                chat = line.replace('\n', '')

                chats.append(chat)
                countries.append(country)

    df = pd.DataFrame(list(zip(countries, chats)),
                      columns=['country', 'chat'])
    return df

def calculate_message_without_bert_topic_label(chats, collection):
    '''
    find new coming scraping data and show how many data should be trained and given topic labels
    Args:
        chats:
        collection:

    '''
    for index, row in chats.iterrows():
        selection_criteria = {
            "$and": [
                {'chat': row['chat']},
                {"topicUpdateTime": {'$exists': False}},
            ],
        }
        projection = {'_id': 1}
        cursor = collection.find(selection_criteria, projection)

        print(len(list(cursor.clone())), "records need to be trained", row['chat'])

def calculate_redundant_embedding(chats, collection):
    '''
    find message that shouldn't have embedding.
    result: message with embedding/message shouldn't have embedding.
    expect 0/xx=0 -> no redundant embedding
    Args:
        chats:
        collection:

    '''
    print("--- Check Redundant Embedding (Expect 0/x)---")
    for index, row in chats.iterrows():
        projection = {'_id': 1}

        selection_criteria = {
            "$and": [
                {'chat': row['chat']},
                {"predicted_class": {'$exists': True}},
                {"messageText": {'$exists': True}},
            ],
            "$or": [
                {"predicted_class": {"$eq": 'Unknown'}},
                {"$expr": {"$lt": [{"$strLenCP": '$messageText'}, 100]}}
            ]
        }

        all_no_embedding_cursor = collection.find(selection_criteria, projection)

        selection_criteria = {
            "$and": [
                {'chat': row['chat']},
                {'embedding': {'$exists': True}},
                {"predicted_class": {'$exists': True}},
                {"messageText": {'$exists': True}},
            ],
            "$or": [
                {"predicted_class": {"$eq": 'Unknown'}},
                {"$expr": {"$lt": [{"$strLenCP": '$messageText'}, 100]}}
            ]
        }

        true_embeding_cursor = collection.find(selection_criteria, projection)

        print("{}/{} of rows have '{}'.  {}".format(
            len(list(true_embeding_cursor.clone())),
            len(list(all_no_embedding_cursor.clone())),
            'embedding', row['chat']))


def calculate_missing_embedding(chats, collection):
    '''
    find message that should have embedding.
    result: message with embedding/message should have embedding.
    expect xx/xx= 1 -> no missing embedding
    Args:
        chats:
        collection:

    '''
    print("--- Check Missing Embedding (Expect x/x) ---")
    for index, row in chats.iterrows():

        projection = {'embedding': 1}

        selection_criteria = {
            "$and": [
                {'chat': row['chat']},
                {"predicted_class": {'$exists': True, "$ne": 'Unknown'}},
                {"messageText": {'$exists': True}},
                {"$expr": {"$gt": [{"$strLenCP": '$messageText'}, 100]}}
            ],
        }
        all_embedding_cursor = collection.find(selection_criteria, projection)


        selection_criteria = {
            "$and": [
                {'chat': row['chat']},
                {'embedding': {'$exists': True}},
                {"predicted_class": {'$exists': True, "$ne": 'Unknown'}},
                {"messageText": {'$exists': True}},
                {"$expr": {"$gt": [{"$strLenCP": '$messageText'}, 100]}}
            ],
        }
        true_embeding_cursor = collection.find(selection_criteria, projection)

        print("{}/{} of rows have '{}'.  {}".format(
            len(list(true_embeding_cursor.clone())),
            len(list(all_embedding_cursor.clone())),
            'embedding', row['chat']))


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def add_embedding(chats, collection):
    '''
    clear embedding which don't meet requirements
        (1) length >100
        (2) bertopic != 'Unknown'
    Args:
        chats:
        collection:

    Returns:

    '''
    import bson
    batch_size = 1000
    for index, row in chats.iterrows():
        print(row['chat'])
        selection_criteria = {
            "$and": [
                {'chat': row['chat']},
                {"predicted_class": {'$exists': True, "$ne": 'Unknown'}},
                {"messageText": {'$exists': True}},
                {"$expr": {"$gt": [{"$strLenCP": '$messageText'}, 100]}}
            ],
        }
        projection = {'_id': 1, 'messageText':1}
        cursor = collection.find_raw_batches(selection_criteria, projection, batch_size=batch_size)

        for batch in cursor:
            data = bson.decode_all(batch)
            df = pd.DataFrame(list(data))
            df['embedding'] = df['messageText'].apply(lambda x: get_embedding(x))

            tmp = list()
            for index, row in df.iterrows():
                tmp.append(UpdateOne({"_id": row["_id"]}, {"$set": {"embedding": row['embedding']}}))
            collection.bulk_write(tmp)

def clear_embedding(chats, collection):
    '''
    clear embedding which don't meet requirements
        (1) length <100
        (2) bertopic == 'Unknown'
    Args:
        chats:
        collection:

    Returns:

    '''

    import bson
    batch_size = 1000
    for index, row in chats.iterrows():
        selection_criteria = {
            "$and": [
                {'chat': row['chat']},
                {"predicted_class": {'$exists': True}},
                {"messageText": {'$exists': True}},
            ],
            "$or": [
                {"predicted_class": {"$eq": 'Unknown'}},
                {"$expr": {"$lt": [{"$strLenCP": '$messageText'}, 100]}}
            ]
        }
        projection = {'_id': 1}
        cursor = collection.find_raw_batches(selection_criteria, projection, batch_size=batch_size)

        for batch in cursor:
            data = bson.decode_all(batch)
            df = pd.DataFrame(list(data))
            # df = df[(df['messageText'].str.len() < 100) | (df['predicted_class'] == 'Unknown')]

            tmp = list()
            for index, row in df.iterrows():
                tmp.append(UpdateOne({"_id": row["_id"]}, {'$unset': {'embedding': 1}}))
            collection.bulk_write(tmp)

def add_model_modification_timestamp(chats, collection):
    modelUpdateTime = datetime.datetime(2023, 10, 17)
    for index, row in chats.iterrows():
        collection.update_many({'chat': row['chat']}, {"$set": {"modelUpdateTime": modelUpdateTime}})

def update_field_name(chats, collection, previous_name, new_name):
    for index, row in chats.iterrows():
        print(row['chat'])
        collection.update_many({'chat': row['chat']}, {"$rename": {previous_name: new_name}})

def update_messageDate(chats, collection):
    '''
    add field 'messageDate' in form of "%Y-%m-%d"
    To accelerate inserting speed, write in bulk
    Args:
        chats:
        collection:

    Returns: add/update messageDate

    '''
    import bson
    batch_size = 1000
    for index, row in chats.iterrows():
        print(row['chat'])

        selection_criteria = {'chat': row['chat']}
        projection = {'_id': 1, 'messageDatetime': 1}
        cursor = collection.find_raw_batches(selection_criteria, projection, batch_size=batch_size)

        for batch in cursor:
            data = bson.decode_all(batch)
            df = pd.DataFrame(list(data))
            df['messageDate'] = df['messageDatetime'].dt.strftime("%Y-%m-%d")

            tmp = list()
            for index, row in df.iterrows():
                tmp.append(UpdateOne({"_id": row["_id"]}, {"$set": {"messageDate": row['messageDate']}}))
            collection.bulk_write(tmp)

def delete_chat_data(chats, collection):
    for index, row in chats.iterrows():
        condition = {'chat': row['chat']}
        collection.delete_many(condition)

if __name__ == '__main__':

    '''
    Add messageDate to the whole collection: scrape.telegram
    use command:
        (1) prd dataset
        python src/database/processScrapeTelegram.py \
        -i data/telegram/queries/switzerland_groups.txt \
        -o scrape.telegram
        
        (2) testing dataset
        python src/database/processScrapeTelegram.py \
        -i data/telegram/queries/switzerland_groups.txt \
        -o test.telegram
        
    '''

    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file_path', help="Specify the input file", type=validate_local_file,
                        required=True)
    parser.add_argument('-o', '--output_database', help="Specify the output database", required=True)
    args = parser.parse_args()

    # connect to db
    ATLAS_TOKEN = os.environ["ATLAS_TOKEN"]
    ATLAS_USER = os.environ["ATLAS_USER"]
    cluster = MongoClient(
        "mongodb+srv://{}:{}@cluster0.fcobsyq.mongodb.net/".format(ATLAS_USER, ATLAS_TOKEN))

    # specify names of database and collection
    # db_name, collection_name = "test", "telegram"
    db_name, collection_name = args.output_database.split('.')
    collection = cluster[db_name][collection_name]

    chats = get_chats_list(args.input_file_path)

    ########### operate collection
    # calculate_message_without_bert_topic_label(chats, collection)

    # clear_embedding(chats, collection)
    # add_embedding(chats, collection)gi
    # calculate_redundant_embedding(chats, collection)
    # calculate_missing_embedding(chats, collection)

    # update_messageDate(chats, collection)

    # replace_empty_state(chats, collection)
    # add_model_modification_timestamp(chats, collection)
    # update_field_name(chats, collection, "modelUpdateTime", "topicUpdateTime")

    ########### operate collection

    cluster.close()

