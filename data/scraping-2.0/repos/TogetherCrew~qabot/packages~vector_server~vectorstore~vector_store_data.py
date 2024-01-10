# import the necessary libraries
import os.path
from datetime import datetime, timedelta
import sys
import json
from langchain.vectorstores import DeepLake
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings

from logger.embedding_logger import logger
from tasks.helper import set_status
from utils import constants
from . import DB_interactions
from .summarize_discord import summarize_discord_main


def main(args):
    # # SET PARAMETERS
    if args is None:
        raise ValueError("No arguments passed to main function.")

    # set openai key
    OA_KEY = args[0]

    # set db information
    DB_CONNECTION_STR = args[1]
    DB_GUILD = args[2]

    task = args[3]

    dates = args[4]
    channels = args[5]
    index_deeplake = args[6]

    logger.debug(f"OA_KEY: {OA_KEY}")
    logger.debug(f"DB_CONNECTION_STR: {DB_CONNECTION_STR}")
    logger.debug(f"DB_GUILD: {DB_GUILD}")

    CHANNELS_ID = ["968110585264898048", "1047205126709969007", "1047205182871707669", "1047390883215052880",
                   "1095278496147849257"] if channels is None else channels
    # DATES = ['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04', '2023-07-05']
    # CHANNELS_ID = ["968110585264898048"]
    DATES = ['2023-10-25', '2023-10-26', '2023-10-27', '2023-10-28', '2023-10-29', '2023-10-30'] if dates is None else dates

    # CHANNELS_ID = [""]
    # DATES = ['2023-04-13', '2023-04-14', '2023-04-15', '2023-04-16', '2023-04-17', '2023-04-18', '2023-04-19']

    # set paths to store results

    # # initiate embeddings model

    # # OpenAI embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=OA_KEY)

    # set_status(task, state='A', meta={'current': 'HF start'})
    # HuggingFace embeddings model
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # embeddings = HuggingFaceEmbeddings(model_name=model_name,client=SentenceTransformer(device='cpu'))

    # set_status(task, state='B', meta={'current': 'HF end'})
    # embed and store data
    vector_store_discord(OA_KEY, DB_CONNECTION_STR, DB_GUILD, CHANNELS_ID, DATES, embeddings, task, index_deeplake)

    return


# # #

def vector_store_discord(OA_KEY, DB_CONNECTION_STR, DB_GUILD, CHANNELS_ID, DATES, embeddings, task, index_deeplake):
    # set up database access
    db_access = DB_interactions.DB_access(DB_GUILD, DB_CONNECTION_STR)
    query = DB_interactions.Query()

    # CHANNELS_ID = list(filter(lambda x: x != "", CHANNELS_ID))

    query_channels = {"channelId": {"$in": list(CHANNELS_ID)}} if len(CHANNELS_ID) > 0 else {}
    set_status(task, state='1', meta={'current': 'MongoDB query'})
    # obtain relations between channel id and name
    cursor = db_access.query_db_find(
        table="channels",
        feature_projection={"__v": 0, "_id": 0, "last_update": 0},
        query=query_channels
    )

    # store relations between channel id and name as dictionary
    channel_id_name = DB_interactions.filter_channel_name_id(list(cursor), channel_name_key="name")

    # CHANNELS_ID = list(channel_id_name.keys())
    # initiate empty doc arrays
    summary_docs = []
    raw_docs = []

    # initiate empty metadata arrays
    all_channels = []
    all_threads = []
    all_authors = []

    set_status(task, state='2', meta={'current': 'Data transforming'})
    total_tokens_per_server = 0
    # for each date
    for date in DATES:

        logger.debug(f"starting date: {date}")

        # compute date before day
        datetime_next_day = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
        date_next_day = datetime_next_day.strftime('%Y-%m-%d')

        set_status(task, state='3', meta={'current': 'Data query'})
        ########## And now querying the table with messages in it ##########

        query_dict = query.create_query_threads(
            channels_id=CHANNELS_ID,
            date_range=[date, date_next_day],
            channelsId_key='channelId',
            date_key='createdDate'
        )

        projection = {
            'user_mentions': 0,
            'role_mentions': 0,
            'reactions': 0,
            'replied_user': 0,
            'type': 0,
            'messageId': 0,
            '__v': 0
        }
        logger.debug(f"query_dict: {query_dict}")
        cursor = db_access.query_db_find(table='rawinfos',
                                         query=query_dict,
                                         feature_projection=projection,
                                         sorting=('datetime', -1)
                                         )

        logger.debug(f"cursor of results")
        # getting a result as thread_results : {str:{str:{str:str}}}
        thread_results = DB_interactions.filter_channel_thread(cursor_list=list(cursor),
                                                               channels_id=CHANNELS_ID,
                                                               thread_id_key='threadId',
                                                               author_key='author',
                                                               message_content_key='content')

        # logger.info("\n\n")
        logger.info(f"thread_results: {thread_results}")
        # logger.info("\n\n")

        set_status(task, state='4', meta={'current': f"Start Summarizing"})
        # run the summarizing function
        logger.debug("Starting summarizing")
        summary_out, num_tokens = summarize_discord_main(thread_results, OA_KEY, True, True)
        logger.debug(f"Finished summarizing: Date: {date} Tokens: {num_tokens}")
        total_tokens_per_server += num_tokens
        logger.debug(f"Until date: {date} Total_Tokens: {total_tokens_per_server}")
        logger.debug(f"Summary_out: {summary_out}")
        set_status(task, state='1B', meta={'current': 'Building Summarize'})
        # add server summary to docs
        summary_docs.append(Document(page_content=summary_out['server_summary']["whole server"],
                                     metadata={
                                         'date': date,
                                         'channel': None,
                                         'thread': None
                                     }))

        # for each channel
        for channel in summary_out['channel_summaries'].keys():

            # store channel summary data
            summary_docs.append(Document(page_content=summary_out['channel_summaries'][channel],
                                         metadata={
                                             'date': date,
                                             'channel': channel_id_name[channel],
                                             'thread': None
                                         }))

            # add channel name to metadata array if it's not in there yet
            if not channel_id_name[channel] in all_channels:
                all_channels.append(channel_id_name[channel])

            # for each thread
            for thread_label in summary_out['thread_summaries'][channel].keys():

                # split thread name
                thread_name_split = thread_label.split(": ")
                thread = thread_name_split[1]

                # store thread summary data
                summary_docs.append(Document(page_content=summary_out['thread_summaries'][channel][thread_label],
                                             metadata={
                                                 'date': date,
                                                 'channel': channel_id_name[channel],
                                                 'thread': thread
                                             }))

                # add thread name to metadata array if it's not in there yet
                if not thread in all_threads:
                    all_threads.append(thread)

                # for each message
                for mess in thread_results[channel][thread].keys():

                    # split message id
                    mess_id_split = mess.split(":")

                    # split author name from handle
                    handle_split = mess_id_split[1].split("#")

                    # if message contains text
                    if len(thread_results[channel][thread][mess]) > 1:

                        # store message
                        raw_docs.append(Document(page_content=thread_results[channel][thread][mess],
                                                 metadata={
                                                     'date': date,
                                                     'channel': channel_id_name[channel],
                                                     'thread': thread,
                                                     'author': handle_split[0],
                                                     'index': mess_id_split[0]
                                                 }))

                        # add author name to metadata array if it's not in there yet
                        if not handle_split[0] in all_authors:
                            all_authors.append(handle_split[0])

    set_status(task, state='H', meta={'current': 'Building DeepLake'})

    PLATFORM_PATH = os.path.join(constants.DEEPLAKE_FOLDER, constants.DEEPLAKE_PLATFORM_FOLDER)
    # check if path exists
    index = 0
    CURRENT_PLATFORM_PATH = f"{PLATFORM_PATH}_{index}"

    if index_deeplake < 0:
        while True:
            logger.debug(f"init CURRENT_PLATFORM_PATH: {CURRENT_PLATFORM_PATH}")
            if os.path.exists(CURRENT_PLATFORM_PATH):
                index += 1
                CURRENT_PLATFORM_PATH = f"{PLATFORM_PATH}_{index}"
                continue
            else:
                logger.debug(f"break CURRENT_PLATFORM_PATH: {CURRENT_PLATFORM_PATH}")
                os.makedirs(CURRENT_PLATFORM_PATH, exist_ok=True)
                break
    else:
        CURRENT_PLATFORM_PATH = f"{PLATFORM_PATH}_{index_deeplake}"

    RAW_DB_SAVE_PATH = os.path.join(CURRENT_PLATFORM_PATH,
                                    constants.DEEPLAKE_RAW_FOLDER)

    SUM_DB_SAVE_PATH = os.path.join(CURRENT_PLATFORM_PATH,
                                    constants.DEEPLAKE_SUMMARY_FOLDER)

    METADATA_OPTIONS_SAVE_PATH = os.path.join(CURRENT_PLATFORM_PATH,
                                              "metadata_options.json")

    # store results in vector stores
    db_raw = DeepLake.from_documents(raw_docs, embeddings, dataset_path=RAW_DB_SAVE_PATH)
    db_summary = DeepLake.from_documents(summary_docs, embeddings, dataset_path=SUM_DB_SAVE_PATH)

    set_status(task, state='I', meta={'current': 'Start write to file'})

    try:
        # store metadata options for vector stores
        JSON_dict = {"all_channels": all_channels, "all_threads": all_threads, "all_authors": all_authors,
                     "all_dates": DATES}

        with open(METADATA_OPTIONS_SAVE_PATH, "w") as outfile:
            json.dump(JSON_dict, outfile)
        set_status(task, state='J', meta={'current': 'END'})
    except BaseException as e:
        logger.error(f"Error on write to file: {e}")
        set_status(task, state='Error', meta={'current': 'END'})
        return
    return


if __name__ == '__main__':
    sys.exit(main(sys.argv))
