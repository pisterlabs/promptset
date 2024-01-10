# Required imports
import datetime
import os
import re

from flask import Flask, request
from firebase_admin import credentials, firestore, initialize_app
import json
from flask_cors import CORS

from diary import Diary
from model import *

# Error codes
ERROR_USER_NOT_FOUND = -1
ERROR_DIARY_DATA_NOT_FOUND = -2

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
user_info_ref = db.collection('userInfo')
user_data_ref = db.collection('userData')
therapist_ref = db.collection('therapist')
therapist_patients_ref = db.collection('therapistPatients')


@app.route("/")
def home():
    return "Hello world!"


@app.route("/test")
def test():
    doc_ref = db.collection(u'todos').document(u'ICMCQGrIbcQszmdhoL7a')

    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        # Cannot return dictionaries in API calls
        return json.dumps(data)
    else:
        return 'No such document!'


# (Home) Get the total amount of diary entries and the frequency of each emotion across all the diary entries
@app.route('/get_diaries_emotion_summary/<uid>')
def get_diaries_emotion_summary(uid):
    entry_counter = 0
    diaries_emotion_summary = {}

    # If user exists, get current user's diary entries data
    if get_diary_entries_ref(uid) is not None:
        diary_entries_data = get_diary_entries_ref(uid).stream()
    else:
        return ERROR_USER_NOT_FOUND

    # Check if user has diary entries
    if diary_entries_data is not None:
        diary_entries = []

        # Move all diary entries to memory
        for entry in diary_entries_data:
            diary = Diary.from_dict(entry.to_dict())
            diary_entries.append(diary)

        # Iterate through each diary entries
        for entry in diary_entries:

            # Get the emotions of each diary entry
            emotions = entry.analysis.get('overallEmotion')

            # Check if the emotions exists
            if emotions is not None:

                # Iterate through each emotion
                for item in emotions:

                    # Add 1 to the total count of each emotion
                    if diaries_emotion_summary.get(item.get('emotion')) is None:
                        diaries_emotion_summary[item.get('emotion')] = 1
                    else:
                        diaries_emotion_summary[item.get('emotion')] += 1

            # Add 1 to the total diary count
            entry_counter += 1

        return json.dumps({'diary_count': entry_counter, 'emotion_summary': diaries_emotion_summary})

    return ERROR_DIARY_DATA_NOT_FOUND


# (Write Diary) Run the content of the diary through the analysis and add new diary entry to database
@app.route('/post_diary', methods=['POST'])
def post_diary():
    data = request.get_json()
    uid = data['uid']
    diary_title = data['title']
    diary_content = data['content']

    # If user exists, get current user's diary entries ref and create a new entry
    if get_diary_entries_ref(uid) is not None:
        diary_entries_ref = get_diary_entries_ref(uid)
        new_entry_ref = diary_entries_ref.document()
    else:
        return json.dumps({"status": ERROR_USER_NOT_FOUND})

    # Remove special characters from diary content before analysis
    diary_content_formatted = re.sub(
        "[!@#$%^&*()_+=.]", "", diary_content, 0, re.IGNORECASE)

    # Run content through analysis
    analysis_results = run_analysis(diary_content_formatted)
    print(analysis_results)

    # Regex out the emotions of the diary from OpenAI response
    analysis_emotions = re.findall('\w+\s\(\d+%\)', analysis_results[0])

    diary_emotions = []

    # Iterate through the emotions and format the data
    for item in analysis_emotions:
        emotion_with_value = re.split('\s', item)

        emotion = emotion_with_value[0].lower()
        value = float(re.sub('[()%]', '', emotion_with_value[1])) / 100

        diary_emotions.append({'emotion': emotion, 'value': value})

    aspects_sentiment = {}
    repeated_aspects = {}
    aspects_sentiment_formatted = []
    overall_sentiment = 0
    diary_sentiment = ''

    print(analysis_results[1])
    # Iterate through the aspect words
    for key, value in analysis_results[1]:
        aspect = key.lower()

        # If aspect is not in aspects_sentiment, add a new aspect entry
        if aspect not in aspects_sentiment.keys():
            aspects_sentiment.update({aspect: float(value)})
        else:

            # If aspect is in aspects_sentiment, update the number of times the aspect has repeated accordingly
            if aspect not in repeated_aspects.keys():
                repeated_aspects.update({aspect: 2})
            else:
                repeated_aspects[aspect] += 1

            # Add up the values with same aspect together
            aspects_sentiment[aspect] += float(value)

    # Iterate through each unique aspect
    for key, value in aspects_sentiment.items():

        # Get the average sentiment score of each unique aspect
        if key in repeated_aspects.keys():
            aspects_sentiment[key] = round(
                (value / repeated_aspects.get(key)), 2)

        # Append average sentiment score to new dict
        aspects_sentiment_formatted.append({'aspect': key, 'value': value})

        # Get the sentiment sum of all the aspects
        overall_sentiment += value

    # Get the average sentiment score of all the aspects
    overall_sentiment /= len(aspects_sentiment)

    # Determine the sentiment of the diary
    if overall_sentiment > 0:
        diary_sentiment = 'positive'
    elif overall_sentiment == 0:
        diary_sentiment = 'neutral'
    else:
        diary_sentiment = 'negative'

    # Create a new diary entry in memory
    new_entry = Diary(uid=new_entry_ref.id,
                      title=diary_title,
                      date=datetime.datetime.now(),
                      content=diary_content,
                      analysis={'aspectSentiment': aspects_sentiment_formatted,
                                'overallEmotion': diary_emotions,
                                'overallSentiment': diary_sentiment})

    # Post diary entry to database
    new_entry_ref.set(new_entry.to_dict())

    return json.dumps({"status": 1})


# (Analysis) Get and format the diaries analysis data from the database
@app.route("/get_analysis_summary/<uid>")
def get_analysis_summary(uid):
    # Get URL query arguments
    date_start = request.args.get('date_start')
    date_end = request.args.get('date_end')

    # If user exists, get current user's diary entries data
    if get_diary_entries_ref(uid) is not None:
        diary_entries_data = get_diary_entries_ref(uid).stream()

    else:
        return ERROR_USER_NOT_FOUND

    analysis_summary = {}
    emotions_over_time = {}
    sentiment_over_time = {}

    # Check if current user have diary entries
    if diary_entries_data is not None:
        diary_entries = []

        # Move all the diary entries to memory
        for entry in diary_entries_data:
            diary = Diary.from_dict(entry.to_dict())
            date = datetime.datetime.strptime(
                diary.date.strftime('%Y-%m-%d'), '%Y-%m-%d')

            # Check if user specified start and end date filter
            if date_start is not None and date_end is not None:

                # Filter and append diaries within a time period
                if datetime.datetime.strptime(date_start, '%Y-%m-%d') <= date <= datetime.datetime.strptime(date_end,
                                                                                                            '%Y-%m-%d'):
                    diary_entries.append(diary)

            else:
                diary_entries.append(diary)

        # Sort diary entries in ascending by date
        diary_entries.sort(key=lambda item: item.date)

        # Iterate through the diaries
        for entry in diary_entries:

            # Get all the unique aspects and append it to a list
            for field in entry.analysis.get('aspectSentiment'):
                aspect = field.get('aspect')

                # To capture error while trying to assign byte type as key instead of string
                try:
                    aspect = aspect.decode('utf-8')
                except AttributeError:
                    pass
                finally:
                    aspect = aspect.lower()

                if aspect not in sentiment_over_time.keys():
                    sentiment_over_time.update({aspect: {}})

            # Get all the unique emotions and append it to a list
            for field in entry.analysis.get('overallEmotion'):
                emotion = field.get('emotion')

                # To capture error while trying to assign byte type as key instead of string
                try:
                    emotion = emotion.decode('utf-8')
                except AttributeError:
                    pass
                finally:
                    emotion = emotion.lower()

                if emotion not in emotions_over_time.keys():
                    emotions_over_time.update({emotion: {}})

        # Iterate through the unique aspects
        for key, value in sentiment_over_time.items():
            repeated_dates = {}

            # Iterate through the diaries
            for entry in diary_entries:

                # Iterate through the diary's aspects
                for field in entry.analysis.get('aspectSentiment'):

                    # Check if aspect in the diary matches the unique aspect
                    if key == field.get('aspect').lower():

                        # Check if the unique aspect already has an entry for that specific date
                        if entry.date.strftime('%Y-%m-%d') in value.keys():

                            # If it does, update in the repeated dates dict
                            if entry.date.strftime('%Y-%m-%d') not in repeated_dates.keys():
                                repeated_dates.update(
                                    {entry.date.strftime('%Y-%m-%d'): 2})
                            else:
                                repeated_dates[entry.date.strftime(
                                    '%Y-%m-%d')] += 1

                            # Sum up the sentiment score of aspect for a specific date
                            value[entry.date.strftime(
                                '%Y-%m-%d')] += field.get('value')
                        else:
                            # Add sentiment score of aspect for a specific date
                            value.update({entry.date.strftime(
                                '%Y-%m-%d'): float(field.get('value'))})

            # Iterate through the dates of each unique aspect and get the average sentiment score
            for date, score in value.items():
                if date in repeated_dates.keys():
                    value[date] = round((score / repeated_dates.get(date)), 2)

        # Iterate through the unique emotions
        for key, value in emotions_over_time.items():
            repeated_dates = {}

            # Iterate through the diaries
            for entry in diary_entries:

                # Iterate through the diary's emotions
                for field in entry.analysis.get('overallEmotion'):

                    # Check if emotion in the diary matches the unique emotion
                    if key == field.get('emotion').lower():

                        # Check if the unique emotion already has an entry for that specific date
                        if entry.date.strftime('%Y-%m-%d') in value.keys():

                            # If it does, update in the repeated dates dict
                            if entry.date.strftime('%Y-%m-%d') not in repeated_dates.keys():
                                repeated_dates.update(
                                    {entry.date.strftime('%Y-%m-%d'): 2})
                            else:
                                repeated_dates[entry.date.strftime(
                                    '%Y-%m-%d')] += 1

                            # Sum up the emotion score of aspect for a specific date
                            value[entry.date.strftime(
                                '%Y-%m-%d')] += field.get('value')
                        else:

                            # Add sentiment score of aspect for a specific date
                            value.update({entry.date.strftime(
                                '%Y-%m-%d'): float(field.get('value'))})

            # Iterate through the dates of each unique emotion and get the average emotion score
            for date, score in value.items():
                if date in repeated_dates.keys():
                    value[date] = round((score / repeated_dates.get(date)), 2)

        # Get the respective total average of each emotion/aspect sentiment of a time period
        emotions_total_average = get_total_average(emotions_over_time)
        sentiment_total_average = get_total_average(sentiment_over_time)

        # debugging
        print(emotions_over_time)
        print(emotions_total_average)
        print(sentiment_over_time)
        print(sentiment_total_average)

        # Compile and format data
        analysis_summary.update({'emotions_over_time': emotions_over_time,
                                 'emotions_total_average': emotions_total_average,
                                 'sentiment_over_time': sentiment_over_time,
                                 'sentiment_total_average': sentiment_total_average})

        return json.dumps(analysis_summary)

    return ERROR_DIARY_DATA_NOT_FOUND


# Get the current user's diaries data if the uid is valid
def get_diary_entries_ref(uid):
    curr_user_data = user_data_ref.document(uid).get()

    if curr_user_data.exists:
        return user_data_ref.document(uid).collection('diaries')
    else:
        return None


# Get the total average score of a time period
def get_total_average(data):
    new_data = {}

    for key, value in data.items():
        sum = 0

        for date, score in value.items():
            sum += score

        new_data.update({key: round((sum / len(value)), 2)})

    return new_data


port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=port, debug=True)
