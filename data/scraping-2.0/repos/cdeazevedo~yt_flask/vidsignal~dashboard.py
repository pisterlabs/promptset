from flask import (
    Blueprint, g, render_template, g, jsonify, current_app
)

import pandas as pd
import openai

from vidsignal.db import get_channels, get_realtime_videos, get_average_views_per_year, get_genres
from . config import OPEN_AI_KEY

app = current_app
bp = Blueprint('dashboard', __name__)
## Routes

@bp.route('/dashboard')
def dashboard():
    # This route is going to pull a bunch of data? Let's start with a list of 
    # channels
    # See if user is logged in or in guest mode
    if g.user:
        logged_in=True
    else:
        logged_in=False
    # get some data dictionary ready
    data = {}
    # Get channel list
    data['channels'] = get_channels()
    
    return render_template('dashboard/dashboard.html', 
                           data=data,
                           logged_in=logged_in,
                           user=g.user)

@bp.route('/dashboard/<selected_channel_id>')
def dashboard_for_channel(selected_channel_id):
    channel_data = {}
    videos = process_realtime_data(get_realtime_videos(selected_channel_id))
    channel_data['realtime'] = videos
    channel_data['upload_frequency'] = upload_frequency(videos)
    channel_data['average_views'] = prepare_data_for_graph(get_average_views_per_year(selected_channel_id))
    # send top 15 realtime videos to openai for summary
    #channel_data['chatGPT_summary'] = get_chatgptsummary(realtime_data[:15])
    return jsonify(channel_data)

@bp.route('/dashboard/genres')
def genre_dashboard():
    genres = get_genres()
    return render_template('dashboard/genres.html', genres=genres)

@bp.route('/dashboard/genres/<genre>')
def return_genre_data(genre):
    return f"<p>{genre}</p>"

## Functions for the dashboard

def process_realtime_data(video_list):
    # """Take the results of the realtime video query from SQL and turn into measure of new views on each video"""
    df = pd.DataFrame(video_list)
    df = df.dropna(how='all')
    df['time_change_seconds'] = df['time_change'].dt.total_seconds()
    df['views_per_day'] = df['views_change'] / (df['time_change_seconds'] / (60 * 60 *24))
    # Convert 'published_date' to datetime objects with the specified format
    df['published_date'] = pd.to_datetime(df['published_date'], format='%Y-%m-%d')
    # Convert datetime objects to ISO 8601 date format (YYYY-MM-DD)
    df['published_date'] = df['published_date'].dt.strftime('%Y-%m-%d')
    data = df.groupby('video_id').agg({
        'views_per_day':'mean',
        'title':'first',
        'duration':'first',
        'published_date':'first',
        'views':'max'
    }).reset_index()
    data.dropna(inplace=True)
    data = data.sort_values(by='views_per_day', ascending=False)
    data = data.to_dict(orient='records')
    return data

def upload_frequency(video_list):
    '''Take in current video list and calculate upload frequency by year/month to chart'''
    df = pd.DataFrame(video_list)
    df['published_date'] = pd.to_datetime(df['published_date'])
    df.set_index('published_date', inplace=True)
    monthly_aggregated = df.resample('M')['video_id'].nunique()
    monthly_aggregated = monthly_aggregated.reset_index()
    monthly_aggregated = monthly_aggregated.rename(columns={'video_id': 'uploads'})
    monthly_aggregated = monthly_aggregated.to_dict(orient='records')
    return monthly_aggregated

import pandas as pd

def prepare_data_for_graph(sql_query_data):
    # Convert the SQL query data into a DataFrame
    df = pd.DataFrame(sql_query_data)
    df['publication_year'] = pd.to_datetime(df['publication_year'], format='%Y', errors='coerce')
    # Set the year as the index
    df.set_index("publication_year", inplace=True)
    # Resample the data to a yearly frequency
    df = df.resample('Y').sum()
    df["average_views"].fillna(0, inplace=True)
    # Reset the index to prepare for graphing
    df.reset_index(inplace=True)
    df.rename(columns={"index": "year"}, inplace=True)
    # Convert the DataFrame to a list of dictionaries
    data_for_graph = df.to_dict(orient='records')
    
    return data_for_graph


## Deal with Openai API
def get_chatgptsummary(video_list):
    """this queries OPEN ai chatgpt and asks for description of videos"""
    openai.api_key = OPEN_AI_KEY
    
    video_titles = [video['title'] for video in video_list]
    video_descriptions = [video['description'] for video in video_list]
    
    prompt = "Write a short answer to the question, what type of videos are these?:\n\n"
    for title, description in zip(video_titles, video_descriptions):
        prompt += f"Title: {title}\nDescription: {description}\n\n"
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150
    )
    
    summary = response.choices[0].text
    
    return response
