from openai import OpenAI
from models import db, Transcript
from configs import UPLOAD_FOLDER

client = OpenAI()

def summary_extraction(transcript):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        n=1,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points. Please use less than 30 words.",
            },
            {"role": "user", "content": transcript},
        ],
    )
    return response.choices[0].message.content


def key_points_extraction(transcript):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        n=1,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about. Please use less than 30 words.",
            },
            {"role": "user", "content": transcript},
        ],
    )
    return response.choices[0].message.content


def action_item_extraction(transcript):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        n=1,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations and extracting action items. Please review the text and identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could be tasks assigned to specific individuals, or general actions that the group has decided to take. Please list these action items clearly and concisely. Please use less than 30 words.",
            },
            {"role": "user", "content": transcript},
        ],
    )
    return response.choices[0].message.content


def sentiment_analysis(transcript):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        n=1,
        messages=[
            {
                "role": "system",
                "content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible. Please use less than 30 words.",
            },
            {"role": "user", "content": transcript},
        ],
    )
    return response.choices[0].message.content


function_map = {
    "keyIdentification": key_points_extraction,
    "actionExtraction": action_item_extraction,
    "summary": summary_extraction,
    "sentiment": sentiment_analysis,
}


def create_task(id: int):
    try:
        print(f"create task {id}")
        from app import app

        with app.app_context():
            transcript = Transcript.query.filter_by(id=id).first()
            # if audio file is provided, convert it to text
            if transcript.audio_name:
                text = audio2text(transcript.audio_name)
                transcript.content = text

            # commit first incase the task is running for a long time
            db.session.commit()

            # run the task
            result = run_task(transcript)

            if not result:
                transcript.status = "Failed"
            else:
                # update the transcript status
                transcript.status = "Finished"

            # save the transcript update
            db.session.commit()
    except Exception as e:
        print(e)
        return False

    return True


def audio2text(audio_name):
    file = open(UPLOAD_FOLDER + "/" + audio_name, "rb")

    if not file:
        return None

    response = client.audio.transcriptions.create(
        file=file,
        model="whisper-1",
        response_format="text",
        language="en"
    )

    return response


# run task and update the transcript
def run_task(transcript: Transcript):
    # check if the task is valid
    if transcript.task not in function_map:
        return False

    transcript.analysis = function_map[transcript.task](transcript.content)

    return True
