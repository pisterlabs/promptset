from string import Template
import json
from datetime import datetime
import pandas as pd
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

MODEL = 'gpt-4-0613'
SQL_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
SUMMARY_DATETIME_FORMAT = "%b %d, %Y"

def execute_query(query, conn):
    cursor = conn.cursor()
    cursor.execute(query)
    df = cursor.fetch_pandas_all()
    cursor.close()
    return df


def get_library_song_name(sp_event, names_dict):
    return names_dict.get(sp_event['ITEM_NAME'], None)


def get_course_name(sp_event, names_dict):
    return names_dict.get(sp_event['COURSE_CONTEXT'], None)

    
def get_lsm_name(sp_event, names_dict):
    return names_dict.get(sp_event['ITEM_NAME'], None)


def get_profile_events(profile_info, queries_path, sql_conn):
    print ('Running profile events query - this might take a while...\n')
    
    profile_log_query = open(queries_path / "get_profile_session_events.sql", "r").read()

    query = Template(profile_log_query).substitute(
        profile_id = profile_info['PROFILE_ID'],
        profile_creation_timestamp = profile_info['PROFILE_CREATION_TIME'],
        )
    
    profile_events_df = execute_query(query, sql_conn)
    
    return profile_events_df
    
def get_summarized_log(events_df, names_dict_path):
    names_dict = json.load(open(names_dict_path, 'r'))
    
    events_df['date'] = events_df['EVENT_TIMESTAMP'].apply(lambda x: str(x).split('.')[0].split(' ')[0])
    events_df['date_formatted'] = events_df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").strftime(SUMMARY_DATETIME_FORMAT))
    events_df['time'] = events_df['EVENT_TIMESTAMP'].apply(lambda x: datetime.strptime(str(x).split('.')[0], SQL_DATETIME_FORMAT))
    events_df['course_name'] = events_df.apply(lambda x: get_course_name(x, names_dict), axis=1)
    events_df['song_name'] = events_df.apply(lambda x: get_library_song_name(x, names_dict), axis=1)
    events_df['lsm_name'] = events_df.apply(lambda x: get_lsm_name(x, names_dict), axis=1)
    
    sessions_df = events_df.groupby('SESSION_ID').agg({'date': ['first'],
                                                       'MINUTES_PLAYED_TOTAL': ['first'],
                                                       'MINUTES_PLAYED_LEVEL': ['first'],
                                                       'MINUTES_PLAYED_STARLEVEL': ['first'],
                                                       'MINUTES_PLAYED_LIBRARY': ['first'],
                                                       'MINUTES_PLAYED_LSM': ['first']}).reset_index()
    
    sessions_df.columns = list([title[0] for title in sessions_df.columns.values])
    
    sessions_df = sessions_df[sessions_df['MINUTES_PLAYED_TOTAL'] > 0]

    session_summary = {}

    for _, session in sessions_df.iterrows():
        played_courses = events_df.loc[events_df['SESSION_ID'] == session['SESSION_ID']]['course_name'].unique()
        played_courses = [course for course in played_courses if course is not None]
            
        library_song_events = events_df.loc[(events_df['SESSION_ID'] == session['SESSION_ID']) &
                                            (events_df['ITEM_TYPE'] == 'library_song')]
        played_library_songs = library_song_events['song_name'].unique()
        played_library_songs = [song for song in played_library_songs if song is not None]
        
        lsm_song_events = events_df.loc[(events_df['SESSION_ID'] == session['SESSION_ID']) &
                                        (events_df['ITEM_TYPE'] == 'lsm_item')]
        played_lsm_songs = lsm_song_events['lsm_name'].unique()
        played_lsm_songs = [song for song in played_lsm_songs if song is not None]
            
        session_summary[session['SESSION_ID']] = {'minutes_played_total': session['MINUTES_PLAYED_TOTAL'],
                                                  'minutes_played_level': session['MINUTES_PLAYED_LEVEL'] + session['MINUTES_PLAYED_STARLEVEL'],
                                                  'minutes_played_library': session['MINUTES_PLAYED_LIBRARY'],
                                                  'minutes_played_lsm': session['MINUTES_PLAYED_LSM'],
                                                  'courses_played': played_courses,
                                                  'library_songs_played': played_library_songs,
                                                  'lsm_songs_played': played_lsm_songs,
                                                  'date': datetime.strptime(session['date'], "%Y-%m-%d").strftime(SUMMARY_DATETIME_FORMAT)}

    summary = ""
    prev_session_date = None
    
    for _, session in session_summary.items():
        summary = session['date'] + ': '
        
        if prev_session_date is not None:
            summary += '%d days from previous session, ' % (datetime.strptime(session['date'], SUMMARY_DATETIME_FORMAT) - 
                                                            datetime.strptime(prev_session_date, SUMMARY_DATETIME_FORMAT)).days
        
        summary += "total time in app: %.01f minutes\n" % (session['minutes_played_total'])
        if session['minutes_played_level'] > 0:
            summary += '%.01f minutes spent in courses: ' % session['minutes_played_level']
            summary += ', '.join(session['courses_played']) + '\n'
            
        if session['minutes_played_library'] > 0:
            summary += '%.01f minutes spent in library songs: ' % session['minutes_played_library']
            summary += ', '.join(session['library_songs_played']) + '\n'
            
        if session['minutes_played_lsm'] > 0:
            summary += '%.01f minutes time spent in Sheet Music (Play or in course): ' % session['minutes_played_lsm']
            summary += ', '.join(session['lsm_songs_played']) + '\n'
            
        summary += '\n'
        
        prev_session_date = session['date']
        
    summary += "%d days have passed since the last session\n" % (datetime.now() - datetime.strptime(sessions_df.iloc[-1]['date'], "%Y-%m-%d")).days
    
    return summary
            

def get_profile_info(profile_id, queries_path, sql_conn):
    
    profile_info_query = open(queries_path / "get_profile_info.sql", "r").read()
    query = Template(profile_info_query).substitute(
        profile_id = profile_id,
        )
    
    def convert_value(value):
        if value is None or type(value) in ('int', 'float', 'str'):
            return value
        else:
            return str(value)
    
    profile_info = {key:convert_value(value[0]) for key, value in execute_query(query, sql_conn).to_dict().items()}
    
    return profile_info


def summarize_profile_activity(summarized_log, profile_info, prompt_path):
    print('Running summarization...\n')
    
    with open(prompt_path, "r") as f:
        prompt = SystemMessage(content=f.read())

    system_info_message_prompt = SystemMessagePromptTemplate.from_template("{profile_info}")
    system_log_message_prompt = SystemMessagePromptTemplate.from_template("{log}")
    
    current_date = datetime.now().strftime(SUMMARY_DATETIME_FORMAT).split(' ')[0]

    chat_prompt = ChatPromptTemplate.from_messages([prompt, system_info_message_prompt, system_log_message_prompt])

    chat = ChatOpenAI(temperature=0.75, model=MODEL)

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    summarized_activity = chain.run(log=summarized_log, current_date=current_date, profile_info=profile_info)

    return summarized_activity

def get_summary(profile_id, summarized_log_path, events_log_path, names_dict_path, 
                queries_path, prompt_path, sql_conn, force_query=False, save_files=True):
    profile_info = get_profile_info(profile_id, queries_path, sql_conn)
    print('Profile info:\n', json.dumps(profile_info, indent=4), '\n')
    
    if not events_log_path.exists() or force_query:
        profile_events_df = get_profile_events(profile_info, queries_path, sql_conn)
        if save_files:
            profile_events_df.to_csv(events_log_path, index=False)
    else:
        profile_events_df = pd.read_csv(events_log_path)
    
    summarized_log = get_summarized_log(profile_events_df, names_dict_path)
    
    if save_files:
        with open(summarized_log_path, "w") as f:
            f.write(summarized_log)
    
    summary = summarize_profile_activity(summarized_log, profile_info, prompt_path)
    
    return summary