from flask import g, current_app
from datetime import datetime, timedelta
from  . config import create_db_connection
import openai


def get_db():
    if 'db' not in g:
        g.db = create_db_connection()
    
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    
    if db is not None:
        db.close()
        
def  init_app(app):
    app.teardown_appcontext(close_db)
    
def get_channels():
    db = create_db_connection()
    cursor = db.cursor()
    sql_query = '''
    SELECT channel_id, name, thumbnail_uri, published_date
    FROM channels c
    WHERE EXISTS (
        SELECT 1
        FROM videos v
        JOIN video_views vv ON v.video_id = vv.video_id
        WHERE v.channel_id = c.channel_id
        GROUP BY v.video_id
        HAVING COUNT(*) >= 2
    )
    ORDER BY name
    '''
    with cursor as crsr:
        crsr.execute(sql_query)
        channels = crsr.fetchall()
        crsr.close()
    # Fetch column names from the cursor description
    column_names = [desc[0] for desc in crsr.description]
    # Convert the list of tuples to a list of dictionaries
    channel_list = [dict(zip(column_names, row)) for row in channels]
    return channel_list

def get_realtime_videos(channel_id):
    """Return a list of videos for a channel to calculate realtime performance."""
    # limit data to timestamps in last 72 hours
    timestamp_limit = datetime.now() - timedelta(hours=72)
    db = create_db_connection()
    cursor = db.cursor()
    sql_query='''
            SELECT
            v.video_id,
            v.title,
            v.duration,
            v.published_date,
            vv.views,
            vv.timestamp,
            TIMEDIFF(vv.timestamp, LAG(vv.timestamp, 1, 0) OVER (PARTITION BY v.video_id ORDER BY vv.timestamp)) AS time_change,
            CASE
                WHEN v.video_id = LAG(vv.video_id, 1) OVER (PARTITION BY v.video_id ORDER BY vv.timestamp) THEN vv.views - LAG(vv.views, 1, 0) OVER (PARTITION BY v.video_id ORDER BY vv.timestamp)
                ELSE NULL
            END AS views_change
        FROM videos v
        LEFT JOIN video_views vv ON v.video_id = vv.video_id
        WHERE v.channel_id = %s AND vv.views > 0 AND vv.timestamp >= %s
        ORDER BY v.video_id, vv.timestamp DESC;
    '''
    with cursor as crsr:
        crsr.execute(sql_query, (channel_id, timestamp_limit))
        videos = crsr.fetchall()
        crsr.close()
    column_names = [desc[0] for desc in crsr.description]
    video_list = [dict(zip(column_names, row)) for row in videos]

    return video_list

def get_average_views_per_year(channel_id):
    db = create_db_connection()
    cursor = db.cursor()
    sql_query='''        
        SELECT YEAR(v.published_date) AS publication_year, AVG(vv.views) AS average_views
        FROM videos v
        LEFT JOIN video_views vv ON v.video_id = vv.video_id
        WHERE v.channel_id = %s AND vv.views > 0
        AND vv.timestamp = (
            SELECT MAX(timestamp) FROM video_views 
            WHERE video_id = v.video_id
        )
        GROUP BY YEAR(v.published_date)
        '''
    with cursor as crsr:
        crsr.execute(sql_query, (channel_id,))
        videos = crsr.fetchall()
        crsr.close()
    column_names = [desc[0] for desc in cursor.description]
    video_list = [dict(zip(column_names, row)) for row in videos]
    return video_list

def get_genres():
    db = create_db_connection()
    cursor = db.cursor()
    sql_query = '''SELECT DISTINCT genre FROM channels ORDER BY genre'''
    cursor.execute(sql_query)
    results = cursor.fetchall()
    cursor.close()
    db.close()

    # Check if there are results before processing
    if results:
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in results]
        return results
    else:
        return []
        