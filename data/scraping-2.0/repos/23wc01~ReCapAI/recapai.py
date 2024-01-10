# Libraries for accessing OpenAI API
import openai
from openai import OpenAI
import os

# In cmd, quick setup of API key by entering: setx KEY_NAME "my_key"
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Returns a dictionary of the summaries
def recap(transcript):
    summary = summarizer(transcript, client)
    sentiment = sentiment_analyzer(transcript, client)
    key_points = key_points_extracter(transcript, client)
    tasks = tasks_extracter(transcript, client)
    return {
        "SENTIMENT": sentiment,
        "SUMMARY": summary,
        "KEY POINTS": key_points,
        "TASKS": tasks
    }
    
# Analyze tone & emotion
def sentiment_analyzer(transcript, client):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are an AI experienced in language and emotion analysis, analyze the sentiment of the following text. Consider overall tone of the discussion, emotion conveyed by language used, and the context in which words and phrases are used. Indicate if sentiment is positvive, negative, or neutral, and provide brief explanations for your analysis where possible."},
            {"role": "user", "content": transcript}
        ]
    )
    return completion.choices[0].message.content

# Summarize into 1 abstract paragraph
def summarizer(transcript, client):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are a highly skilled AI trained in language comprehension and sumarization. Read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, give a coherent and readable summary that could help a person understand the main points of the disscussion without needing to read the entire text. Avoid unnecessary details or tangential points."},
            {"role": "user", "content": transcript}
        ]
    )
    return completion.choices[0].message.content

# List key points
def key_points_extracter(transcript, client):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are a proficient AI that distills information into key points. Based on the following text, identify and list the main points that were discussed. These should be the most important ideas, finidngs, or topics that are crucial to the essence of the discussion. Your goal is to provide a list someone could scan over quickly and understand what was talked about."},
            {"role": "user", "content": transcript}
        ]
    )
    return completion.choices[0].message.content

# List action items (assignments, tasks, actions, etc.)
def tasks_extracter(transcript, client):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are an AI expert in analyzing conversations and extracting action items. Please review text and identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could be tasks assigned to specific individuals, or general actions that the group has decided to take. List action items clearly and concisely."},
            {"role": "user", "content": transcript}
        ]
    )
    return completion.choices[0].message.content


    












         

