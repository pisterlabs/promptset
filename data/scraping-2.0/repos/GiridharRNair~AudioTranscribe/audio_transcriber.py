import os
import openai
import tempfile
from datetime import date
from pydub import AudioSegment
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")
email_sender = SendGridAPIClient(os.getenv("SENDGRID_KEY"))


def process_transcription(upload_path, email):
    try:
        full_transcript = transcribe(upload_path)

        if not full_transcript:
            raise Exception("Transcription failed")

        summary = meeting_minutes(full_transcript)

        if not summary:
            raise Exception("Summary generation failed")

        send_email(email, summary, full_transcript)

    except Exception as e:
        print(f"Error in transcription process: {str(e)}")


def transcribe(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        max_segment_size = 20 * 500

        with tempfile.TemporaryDirectory() as temp_dir:
            segments = []
            transcriptions_segments = []

            start_time = 0
            while start_time < len(audio):
                end_time = min(start_time + max_segment_size, len(audio))
                segment = audio[start_time:end_time]
                temp_file_path = os.path.join(temp_dir, f"segment_{start_time}-{end_time}.wav")
                segment.export(temp_file_path, format="wav")
                segments.append(temp_file_path)
                start_time = end_time

            for segment_path in segments:
                with open(segment_path, 'rb') as segment_file:
                    transcription = openai.Audio.transcribe("whisper-1", segment_file)

                transcriptions_segments.append(transcription['text'])
                os.remove(segment_path)

            os.remove(file_path)
            return ' '.join(transcriptions_segments)

    except Exception as e:
        return str(e)


def send_email(recipient, transcription_info, transcript):
    transcript_summary = f"Summary: \n{transcription_info['abstract_summary']} \n\n"
    key_points = f"Key Points: \n{transcription_info['key_points']} \n\n"
    action_items = f"Action Items: \n{transcription_info['action_items']} \n\n"
    transcript_sentiment_analysis = f"Sentiment Analysis: \n{transcription_info['sentiment']}\n\n"
    full_transcript = f"Full Transcript:\n{transcript}"
    try:
        subject = f'Transcript - {str(date.today())}'
        message = Mail(
            from_email='talktotextpro@gmail.com',
            to_emails=recipient,
            subject=subject,
            plain_text_content=transcript_summary+key_points+action_items+transcript_sentiment_analysis+full_transcript
        )
        email_sender.send(message)
        print("Transcription successful")
    except Exception as e:
        print(str(e))


def meeting_minutes(transcription):
    abstract_summary = abstract_summary_extraction(transcription)
    key_points = key_points_extraction(transcription)
    action_items = action_item_extraction(transcription)
    sentiment = sentiment_analysis(transcription)
    return {
        'abstract_summary': abstract_summary,
        'key_points': key_points,
        'action_items': action_items,
        'sentiment': sentiment
    }


def split_text_into_chunks(text, chunk_size=3000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks


def abstract_summary_extraction(transcription):
    chunks = split_text_into_chunks(transcription)
    summaries = []

    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a highly skilled AI trained in language comprehension and "
                                              "summarization. I would like you to read the following text and summarize"
                                              "it into a concise abstract paragraph. Aim to retain the most important "
                                              "points, providing a coherent and readable summary that could help a "
                                              "person understand the main points of the discussion without needing to "
                                              "read the entire text. Please avoid unnecessary details or tangential "
                                              "points."},
                {"role": "user", "content": chunk}
            ]
        )
        summaries.append(response['choices'][0]['message']['content'])

    return ' '.join(summaries)


def key_points_extraction(transcription):
    chunks = split_text_into_chunks(transcription)
    key_points = []

    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a proficient AI with a specialty in distilling information into "
                                              "key points. Based on the following text, identify and list the main "
                                              "points that were discussed or brought up. These should be the most "
                                              "important ideas, findings, or topics that are crucial to the essence of "
                                              "the discussion. Your goal is to provide a list that someone could read "
                                              "to quickly understand what was talked about."},
                {"role": "user", "content": chunk}
            ]
        )
        key_points.append(response['choices'][0]['message']['content'])

    return ' '.join(key_points)


def action_item_extraction(transcription):
    chunks = split_text_into_chunks(transcription)
    action_items = []

    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are an AI expert in analyzing conversations and extracting action "
                                              "items. Please review the text and identify any tasks, assignments, or "
                                              "actions that were agreed upon or mentioned as needing to be done. These "
                                              "could be tasks assigned to specific individuals, or general actions "
                                              "that the group has decided to take. Please list these action items "
                                              "clearly and concisely."},
                {"role": "user", "content": chunk}
            ]
        )
        action_items.append(response['choices'][0]['message']['content'])

    return ' '.join(action_items)


def sentiment_analysis(transcription):
    chunks = split_text_into_chunks(transcription)
    sentiments = []

    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "As an AI with expertise in language and emotion analysis, your task is "
                                              "to analyze the sentiment of the following text. Please consider the "
                                              "overall tone of the discussion, the emotion conveyed by the language "
                                              "used, and the context in which words and phrases are used. Indicate "
                                              "whether the sentiment is generally positive, negative, or neutral, and "
                                              "provide brief explanations for your analysis where possible."},
                {"role": "user", "content": chunk}
            ]
        )
        sentiments.append(response['choices'][0]['message']['content'])

    return ' '.join(sentiments)
