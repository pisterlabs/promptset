import boto3
from openai import OpenAI
import json

def handler(event, context):
    s3_client = boto3.client("s3")
    s3_client.download_file("dr-watson-patient-recordings",
                            f"{event['pathParameters']['patientId']}.mp3", "/tmp/patient.mp3")

    client = OpenAI(
        api_key="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")

    audio_file = open("patient.mp3", "rb")
    transcript = client.audio.translations.create(
        model="whisper-1",
        file=audio_file
    )
    print(transcript.text)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an intern in the healthcare domain and have been tasked to take a patient's medical history."},
            {"role": "user", "content": f"This is what the patient has said so far '{transcript.text}' Now based on this patient's medical history, can you give me a list of questions i have to ask to the patient to get more information from them, so that doctor has all the information to help the patient. Give me the questions in the following format 'question#: question' Don't say anything other than the above format. don't format the text . give me only 3 follow up questions"},
        ]
    )

    print(response.choices[0].message.content)
    return {
        "statusCode": 200,
        "body": json.dumps({
            "questions": response.choices[0].message.content
        })
    }
