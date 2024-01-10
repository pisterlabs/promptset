import urllib.request as urllib2
import ssl
import boto3
from moviepy.editor import VideoFileClip
import uuid
import time
from botocore.exceptions import NoCredentialsError
import pandas as pd
import os
from openai import OpenAI



## Credentials to AWS and Other API Services
aws_access_key_id = '<MASKED>'
aws_secret_access_key = '<MASKED>'
region_name = '<MASKED>'
bucket_name = '<MASKED>'
queue_url = '<MASKED>'
chat_gpt_api_key = "<MASKED>"


def translate_text(text, source_language, target_language):
    translate_client = boto3.client('translate', aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key,region_name=region_name)

    source_language_code = source_language
    target_language_code = target_language

    response = translate_client.translate_text(
        Text=text,
        SourceLanguageCode=source_language_code,
        TargetLanguageCode=target_language_code
    )
    
    translated_text = response['TranslatedText']
    return translated_text


def extract_audio(video_path, output_audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio

    audio_clip.write_audiofile(output_audio_path, codec='mp3')
    video_clip.close()



if __name__ == "__main__":

    while True:
        try:
            sqs = boto3.client('sqs', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
            s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
            transcribe = boto3.client('transcribe', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key,region_name=region_name)


            response = sqs.receive_message(
                QueueUrl=queue_url,
                AttributeNames=['All'],
                MaxNumberOfMessages=1,
                MessageAttributeNames=['All'],
                VisibilityTimeout=0,
                WaitTimeSeconds=0
                )
            
            if 'Messages' in response:
                ## Finding Session ID From SQS Message
                message = response['Messages'][0]
                receipt_handle = message['ReceiptHandle']
                print(f"Received message: {message['Body']}")
                
                session_id = message['Body']
                print("Session ID - {}".format(session_id))

                ## Download Video From OpenVidu Deployment Server
                dwn_link = 'https://<MASKED>/openvidu/recordings/{}/{}.mp4'.format(session_id, session_id)
                context = ssl._create_unverified_context()
                print('Video Recording Link: {}'.format(dwn_link))

                video_name = session_id+'.mp4' 
                rsp = urllib2.urlopen(dwn_link, context=context)
                with open(video_name,'wb') as f:
                    f.write(rsp.read())
                print("Video file name: {}".format(video_name))

                ## Create folder for the Session_ID in the S3 bucket
                s3.put_object(Bucket=bucket_name,Body='', Key='{}/'.format(session_id))
                print("created folder {} in {} bucket".format(session_id, bucket_name))

                try:
                    s3.upload_file(video_name, bucket_name, '{}/{}.mp4'.format(session_id, session_id))
                    print(f"Video uploaded successfully to S3: {'{}/{}.mp4'.format(session_id, session_id)}")
                except Exception as e:
                    print(f"Error uploading file to S3: {e}")


                ## Extracting Audio from the video file
                audio_name = session_id+'.mp3' 
                extract_audio(video_name, audio_name)
                print('extracted audio: {}'.format(audio_name))

                try:
                    s3.upload_file(audio_name, bucket_name, '{}/{}.mp3'.format(session_id, session_id))
                    print(f"Audio uploaded successfully to S3: {'{}/{}.mp3'.format(session_id, session_id)}")
                except Exception as e:
                    print(f"Error uploading file to S3: {e}")


                ## Get English Transcripts
                #transcript_name = session_id+'_transcript.txt'
                transcribe_job = str(uuid.uuid4())
                print("Transcribe Job: {}".format(transcribe_job))
                transcribe.start_transcription_job(
                    TranscriptionJobName=transcribe_job,
                    Media={'MediaFileUri': 's3://{}/{}/{}.mp3'.format(bucket_name, session_id, session_id)},
                    MediaFormat='mp3',
                    LanguageCode='en-US',
                    Settings = {
                        'ShowSpeakerLabels': True,
                        'MaxSpeakerLabels': 10
                        }
                        )
                

                i = 0
                transcription_result = dict()
                while True:
                    transcription_result = transcribe.get_transcription_job(TranscriptionJobName=transcribe_job)
                    print("Transcribe Job Status: {}".format(transcription_result['TranscriptionJob']['TranscriptionJobStatus']))
                    if transcription_result['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                        break
                    time.sleep(15)
                    i+=1
                    if i==20:
                        break

                

                transcription_result = transcribe.get_transcription_job(TranscriptionJobName=transcribe_job)
                df = pd.read_json(transcription_result['TranscriptionJob']['Transcript']['TranscriptFileUri'])

                spk = 'dummy'
                convs = []
                temp = ''
                for tok in df.loc['items', 'results']:
                    if tok['speaker_label'] != spk:
                          convs.append([spk, temp])
                          temp = ''
                          spk = tok['speaker_label']
                          temp = ' '.join([temp, tok['alternatives'][0]['content']])
                    else:
                      temp = ' '.join([temp, tok['alternatives'][0]['content']])
                
                convs.append([spk, temp])
                convs = convs[1:]


                english_convs = convs
                spanish_convs = []
                chinese_convs = []
                hindi_convs = []
                german_convs = []

                for conv in english_convs:
                     spanish_convs.append([conv[0], translate_text(conv[1], 'en', 'es')])
                     chinese_convs.append([conv[0], translate_text(conv[1], 'en', 'zh')])
                     hindi_convs.append([conv[0], translate_text(conv[1], 'en', 'hi')])
                     german_convs.append([conv[0], translate_text(conv[1], 'en', 'de')])
                    
                
                
                ## Language Translations
                # Upload English Transcripts
                transcript_name = "english_transcript.txt"
                f = open(transcript_name, "w")
                for conv in english_convs:
                     f.write(conv[0]+' : '+conv[1])
                     f.write("\n\n")
                f.close()

                try:
                    s3.upload_file(transcript_name, bucket_name, '{}/{}_{}'.format(session_id, session_id, transcript_name))
                    print(f"Transcript uploaded successfully to S3: {'{}/{}_{}'.format(session_id, session_id, transcript_name)}")
                except Exception as e:
                    print(f"Error uploading file to S3: {e}")

                # Upload Spanish Transcripts
                transcript_name = "spanish_transcript.txt"
                f = open(transcript_name, "w")
                for conv in spanish_convs:
                    f.write(conv[0]+' : '+conv[1])
                    f.write("\n\n")
                f.close()

                try:
                    s3.upload_file(transcript_name, bucket_name, '{}/{}_{}'.format(session_id, session_id, transcript_name))
                    print(f"Transcript uploaded successfully to S3: {'{}/{}_{}'.format(session_id, session_id, transcript_name)}")
                except Exception as e:
                    print(f"Error uploading file to S3: {e}")


                # Upload Chinese Transcripts
                transcript_name = "chinese_transcript.txt"
                f = open(transcript_name, "w")
                for conv in chinese_convs:
                    f.write(conv[0]+' : '+conv[1])
                    f.write("\n\n")
                f.close()

                try:
                    s3.upload_file(transcript_name, bucket_name, '{}/{}_{}'.format(session_id, session_id, transcript_name))
                    print(f"Transcript uploaded successfully to S3: {'{}/{}_{}'.format(session_id, session_id, transcript_name)}")
                except Exception as e:
                    print(f"Error uploading file to S3: {e}")


                # Upload Hindi Transcripts
                transcript_name = "hindi_transcript.txt"
                f = open(transcript_name, "w")
                for conv in hindi_convs:
                    f.write(conv[0]+' : '+conv[1])
                    f.write("\n\n")
                f.close()

                try:
                    s3.upload_file(transcript_name, bucket_name, '{}/{}_{}'.format(session_id, session_id, transcript_name))
                    print(f"Transcript uploaded successfully to S3: {'{}/{}_{}'.format(session_id, session_id, transcript_name)}")
                except Exception as e:
                    print(f"Error uploading file to S3: {e}")

                
                # Upload German Transcripts
                transcript_name = "german_transcript.txt"
                f = open(transcript_name, "w")
                for conv in german_convs:
                    f.write(conv[0]+' : '+conv[1])
                    f.write("\n\n")
                f.close()

                try:
                    s3.upload_file(transcript_name, bucket_name, '{}/{}_{}'.format(session_id, session_id, transcript_name))
                    print(f"Transcript uploaded successfully to S3: {'{}/{}_{}'.format(session_id, session_id, transcript_name)}")
                except Exception as e:
                    print(f"Error uploading file to S3: {e}")


                

                ## ChatGPT Integration
                client = OpenAI(api_key=chat_gpt_api_key)

                f = open('english_transcript.txt', "r+")
                txt = f.read()


                prompt = """
                Prompt is as follows:
                
                [SENTIMENT]
                Analyze the sentiment of the given call transcript.

                [TOPIC MODELING]
                Perform topic modeling on the call transcript and identify key themes.

                [SPEAKER CONTRIBUTION]
                Analyze the contribution of each speaker in terms of word count.

                [KEYWORD EXTRACTION]
                Extract keywords or phrases that are frequently mentioned in the call transcript.

                [NAMED ENTITY RECOGNITION]
                Identify and extract named entities (people, organizations, locations) from the call transcript.

                [DIALOGUE FLOW ANALYSIS]
                Analyze the flow of the conversation, including turn-taking patterns and notable shifts in topic.

                [DIVERSITY OF OPINIONS]
                Assess instances where speakers express agreement or disagreement, highlighting the diversity of opinions.

                [KEY QUOTES IDENTIFICATION]
                Identify and extract key quotes or statements made by the speakers.

                [CONTEXTUAL ANALYSIS]
                Perform contextual analysis to understand the broader context surrounding certain statements and discussions.

                [LANGUAGE COMPLEXITY]
                Assess the complexity of language used in the call transcript.
                
                [END]
                """

                user_message = "Analyze the following text transcript of a conversation. The output must be professional report with title as ANALYSIS OF THE CALL. It should look like a Professional report which can be output directly to a user and should not contains any sentences that are like an interaction between a chat assistant and a user. Also, \n means its end of sentence. The Transcript is as  follows:  " + txt + "\n \n" + prompt


                chat_completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": user_message}]
                    )
                

                transcript_name = 'cgpt_analysis.txt'
                f = open(transcript_name, 'w')
                f.write(chat_completion.choices[0].message.content)
                f.close()


                try:
                    s3.upload_file(transcript_name, bucket_name, '{}/{}_{}'.format(session_id, session_id, 'cgpt_analysis.txt'))
                    print(f"ChatGPT Analysis uploaded successfully to S3: {'{}/{}_{}'.format(session_id, session_id, 'cgpt_analysis.txt')}")
                except Exception as e:
                    print(f"Error uploading file to S3: {e}")



                # Delete received message from queue
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle
                    )

                try:
                    os.remove(video_name)
                    os.remove(transcript_name)
                    os.remove(audio_name)
                    os.remove('english_transcript.txt')
                    os.remove('spanish_transcript.txt')
                    os.remove('chinese_transcript.txt')
                    os.remove('hindi_transcript.txt')
                    os.remove('german_transcript.txt')
                    os.remove('{}/{}_{}'.format(session_id, session_id, 'cgpt_analysis.txt'))
                except:
                    pass

            else:
                print("No messages in the queue")
        

        except NoCredentialsError:
            print("Credentials not available")
        
        time.sleep(10)