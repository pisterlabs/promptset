import botocore
import openai
import whisper
from celery import shared_task
from v1.models import transcript
import nltk.data
import boto3

openai.api_key = ""
model = whisper.load_model('tiny.en')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


@shared_task
def transcribe_audio(key, instance_id):
    t = transcript.objects.get(id=instance_id)

    session = boto3.session.Session()
    client = session.client('s3',
                            config=botocore.config.Config(s3={'addressing_style': 'virtual'}),
                            region_name = 'sfo3',
                            endpoint_url = 'https://sfo3.digitaloceanspaces.com',
                            aws_access_key_id = "",
                            aws_secret_access_key = "")

    url = client.generate_presigned_url(ClientMethod='get_object', Params={'Bucket': 'whisperapp', 'Key': key}, ExpiresIn=3600)


    result = model.transcribe(url)
    t.result_raw = result['text']
    t.save()
    process_transcription.delay(instance_id)


@shared_task
def process_transcription(instance_id):
    t = transcript.objects.get(id=instance_id)
    t.result_processed = "<br>".join(tokenizer.tokenize(t.result_raw))
    t.save()
