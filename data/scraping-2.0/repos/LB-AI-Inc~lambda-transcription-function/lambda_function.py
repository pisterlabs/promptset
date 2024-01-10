import json
import requests
import urllib
import openai
import boto3
import os
import re
import time
import subprocess


def replace_ssns(string):
    ssn_pattern = r'\b(\d{3}-\d{2}-\d{4}|\d{9})\b'
    result_string = re.sub(ssn_pattern, 'XXX-XX-XXXX', string)
    return result_string

def replace_credit_card_numbers(input_string):
    card_number_pattern = r'\b(\d{4}-?\d{4}-?\d{4}-?\d{4}|\d{16})\b'
    result_string = re.sub(card_number_pattern, 'XXXXXXXXXXXXXXXX', input_string)
    return result_string

def replace_4_digits(string):
    four_digit_pattern = r'\b\d{4}\b'
    result_string = re.sub(four_digit_pattern, 'XXXX', string)
    return result_string

def replace_digits_with_x(string):
    result_string = ''.join(['X' if char.isdigit() else char for char in string])
    return result_string

def move_file(source_bucket, source_key, destination_bucket, destination_key):
    s3 = boto3.client('s3')
    try:
        # Copy the object
        s3.copy_object(
            Bucket=destination_bucket,
            Key=destination_key,
            CopySource={
                'Bucket': source_bucket,
                'Key': source_key
            }
        )

        # Delete the original object
        s3.delete_object(
            Bucket=source_bucket,
            Key=source_key
        )

        print(f"File moved from {source_bucket}/{source_key} to {destination_bucket}/{destination_key}")
        return True
    except Exception as e:
        print(f"Error moving file: {e}")
        return False

def with_function(question, system_prompt, transcript):
    functions = [{
            'name': 'answer_generator',
            'description': 'Get an answer to the question',
            'parameters': {
                'type': 'object',
                'properties': {
                    'answer': {
                        'type': 'boolean',
                        'description': question,
                    },
                    'reasoning': {
                        'type': 'string',
                        'description': 'Explain your reasoning.'
                    }
                }
            }
        }]

    response = openai.ChatCompletion.create(
        engine="GPTturbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcript
            }
        ],
        functions=functions,
        function_call={
            "name": "answer_generator",
        },
    )
    return response

def without_function(question, system_prompt, transcript):
    response = openai.ChatCompletion.create(
            engine="GPTturbo",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": transcript + "\n\nQuestion: " + question
                }
            ]
        )
    return response

def get_openai_credentials():
    secret_name = "openAI_credentials"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)

    # Retrieve the secret
    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response['SecretString'])

        # Extract OpenAI and WhisperAI credentials
        openai_credentials = {
            'api_key': secret.get('OpenAI', ''),
            'api_base': "https://lightbulb.openai.azure.com/",
            'api_version': "2023-07-01-preview"
        }
        whisperai_credentials = {
            'api_key': secret.get('WhisperAI', ''),
            'api_base': "https://lightbulb-whisper.openai.azure.com/",
            'api_version': "2023-09-01-preview"
        }
        return openai_credentials, whisperai_credentials

    except Exception as e:
        print(f"Error retrieving OpenAI and WhisperAI credentials from Secrets Manager: {e}")
        return None, None


def get_audio_duration(file_path):
    try:
        # Command to get audio duration using FFMPEG
        output_file = '/tmp/output.mp3'  # Output file path
        p1 = subprocess.run(["/opt/ffmpeg", "-i", file_path, output_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        duration_pattern = re.compile(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d+)")
        duration_match = duration_pattern.search(p1.stderr.decode("utf-8"))

        if duration_match:
            hours, minutes, seconds = map(float, duration_match.groups())
            total_seconds = (hours * 3600) + (minutes * 60) + seconds
            os.remove(output_file)
            return round(total_seconds)
        else:
            return None

    except subprocess.CalledProcessError as e:
        # Handle FFMPEG command execution errors
        print("FFMPEG error:", e)
        return None

def analyze(transcript, openai_credentials):
    # initialize the chat completion azure creds
    api_key, api_base, api_version = openai_credentials['api_key'], openai_credentials['api_base'], openai_credentials['api_version']
    openai.api_key = api_key
    openai.api_base = api_base
    openai.api_version = api_version

    # system_prompt = "You are an expert sales coach. You are helping critique a call for people to gather data and help people improve. Answer the questions as asked and provide clear details when asked for. The transcript is provided and be clear that there could be multiple people on the call. Do not include any Social Secuirty numbers in your response, this is sensitive information."
    system_prompt = "You are an expert quality assurance manager for an inbound call center. You are helping critique inbound calls made by customers, who we’ll refer to as participants, to our company representatives. Answer the questions as asked and provide clear details when asked for. The transcript of the call is provided and there are multiple people on the call- the participant (the customer) and the representative of the company (whose performance you are evaluating). On all calls, reps are required to verify the Social Security number of the caller for security reasons. Do not criticize the rep for asking for sensitive information like this, as it is a part of the standard process."

    promptList = {
        "DepartmentID": {
            "type": "DepartmentID",
            "question": "Was this call related to NueSynergy or PlanSource? Answer should be either PlanSource or NueSynergy.",
            "graded": False,
        },
        "RepID": {
            "type": "RepID",
            "question": "What was the name of the representative who answered the phone call?",
            "graded": False,
        },
        "CustomerID": {
            "type": "CustomerID",
            "question": "What was the name of the customer who called in asking for help?",
            "graded": False,
        },
        "Greeting": {
            "type": "Greeting",
            "question": "Did the representative greet the participant by saying something like 'Thank you for calling NueSynergy' or 'PlanSource, my name is (Their Name)' or something similar to this? Another example would be 'Plansource or Nuesynergy, This is (THEIR NAME) how can I help you?' The rep must say their name and NueSynergy/PlanSource in their greeting. to get credit for this. If the rep introduced themselves by stating their company name, and their personal name, that is a positive answer. If they did not say their name and either PlanSource or NueSynergy, that is a negative answer.",
            "graded": True,
        },
        "SecurityVerification": {
            "type": "SecurityVerification",
            "question": "Did the participant disclose their social security number? Either the full number or their last four digits counts as a positive answer. If they didn’t verify the social, did they verify the last four digits? If the answer to all of those is no, it's a negative answer. Do not repeat the number when stating your reasoning.",
            "graded": True,
        },
        "NeedsAssessment": {
            "type": "NeedsAssessment",
            "question": "Did the representative ask, 'How can I help you today?' or something similar to this? If they did ask this or something very similar, this is a positive answer. If they did not ask a question to understand why the client called, this is a negative answer.",
            "graded": True,
        },
        "RepDiscovery": {
            "type": "RepDiscovery",
            "question": "Did the representative ask relevant questions to further understand the reasoning behind why the customer contacted them? It's ok if they verified the customer first. If they did, this is a positive answer. If they didn’t, this is a negative answer.",
            "graded": True,
        },
        "ResourceUtilization": {
            "type": "ResourceUtilization",
            "question": "Does it appear that the representative used all the available resources they have at their disposal to resolve the customer’s issue? If they are taking a long time to answer the question, not seeming confident in their answers, or the customer is confused, this is likely evidence that they should have utilized additional resources to answer the call. If they did use all available resources, this is a positive answer. If the issue was resolved on this call, and it isn’t obvious that the rep was taking a long time to answer, not seeming confident, or the customer wasn’t confused by the rep’s responses, this is also a positive answer. If the rep did take a long time to answer questions, or wasn’t confident, or confused the customer, this is a negative answer.",
            "graded": True,
        },
        "Responsiveness": {
            "type": "Responsiveness",
            "question": "Did the representative provide the participant with thorough and accurate responses to all questions? If they did, this is a positive answer. If they did not, this is a negative answer. Explain your reasoning.",
            "graded": True,
        },
        "HoldCompliance": {
            "type": "HoldCompliance",
            "question": "Does it appear the rep placed the customer on hold? Evidence of this would be the rep saying something like 'I’m going to place you on a brief hold.' If the word ‘hold’ was never mentioned, and it’s clear the rep didn’t place the customer on hold, you should answer true. If the rep did mention placing the customer on hold, you should answer false.",
            "graded": True,
        },
        "TransferCompliance": {
            "type": "TransferCompliance",
            "question": "Did the representative mention they would be transferring the participant to to someone else to help them? Evidence of this would include the rep saying something like “I’m going to transfer you to” If the word ‘transfer’ was never mentioned, and it’s clear the rep didn’t transfer the customer to another department, you should answer true. If the rep did mention transferring the customer, you should answer false.",
            "graded": True,
        },
        "CallDisconnected": {
            "type": "CallDisconnected",
            "question": "Was this call disconnected unintentionally? Evidence of this would include the call ending abruptly, without a formal goodbye. If there was no formal goodbye, you should answer false. If there was a formal goodbye and the call did not end abruptly, you should answer true.",
            "graded": True,
        },
        "RecapConfirmation": {
            "type": "RecapConfirmation",
            "question": "Did the representative provide a recap of the participant’s issue along with the resolution? Evidence of this would include the representative discussing next steps, or actions they will take to resolve the customer’s concern. If the representative did clarify how they are going to solve the customer’s issue, this is a positive answer. If they did not provide any kind of summary of the call, or the solution or steps the rep was going to take to solve the issue, this is a negative answer.",
            "graded": True,
        },
        "AdditionalSupportCheck": {
            "type": "AdditionalSupportCheck",
            "question": "Did the representative or the participant indicate that all questions were answered or all needs were met before ending the call? If the representative asked a question like 'Is there anything else I can assist you with?' or if the participant indicated their needs were met, this is a positive answer. If neither occurred, this is a negative answer.",
            "graded": True,
        },
        "SurveyRequest": {
            "type": "SurveyRequest",
            "question": "Did the representative ask if the participant would like to stay on the line to take a brief survey to provide feedback on their call? Or did they say “please hold for a brief survey.” If they did ask about or mention the survey, this is a positive answer. If they did not ask about or mention the survey at the end of the call, this is a negative answer.",
            "graded": True
        },
        "ClosingFormality": {
            "type": "ClosingFormality",
            "question": "Did the representative complete the call with a proper closing? An example of this would include the rep saying something like 'Thank you for calling and have a great day!' If the rep did thank them for calling and said goodbye, this is a positive answer. If the rep did not say thank you for calling and said goodbye, this is a negative answer.",
            "graded": True
        },
        "CourtesyAssessment": {
            "type": "CourtesyAssessment",
            "question": "Did the representative treat the participant with courtesy? Evidence of this would include the rep consistently addressing the participant by their name, showing empathy, and being professional. If they were generally courteous and professional, this is a positive answer. If the rep never called the participant by name, or was unprofessional or rude, this is a negative answer.",
            "graded": True
        },
        "CommunicationTone": {
            "type": "CommunicationTone",
            "question": "Based on the language used by the rep, how positive was their tone and sentiment? Did they display an engaged, pleasant tone? If they They maintained a professional and courteous demeanor throughout the call, this is a positive answer. If the rep was rude or unprofessional, this is a negative answer.",
            "graded": True
        },
        "ListeningEtiquette": {
            "type": "ListeningEtiquette",
            "question": "Did the representative wait for the participant to finish talking before speaking? If they interrupt in error, did they apologize for the interruption?  If there were no interruptions and the rep displayed proper listening etiquette, or if they apologized for interrupting, this is a positive answer. If the rep interrupted the participant and did not apologize for interrupting, this is a negative answer.",
            "graded": True
        },
        "EmpathyAssessment": {
            "type": "EmpathyAssessment",
            "question": "Did the representative treat the participant with respect by trying to understand the cause of the participant’s problem, understand how their problem affects their behavior, and respond with empathy? If yes, this is a positive answer. If no, this is a negative answer.",
            "graded": True
        },
        "HonestyAssessment": {
            "type": "HonestyAssessment",
            "question": "Did the rep give accurate answers in a confident tone that addressed the concern of the customer directly?  If yes, this is a positive answer. If no, this is a negative answer.",
            "graded": True
        },
        "InterestEngagement": {
            "type": "InterestEngagement",
            "question": "Did the representative exhibit interest in the participant's questions or concerns?  If yes, this is a positive answer. If no, this is a negative answer.",
            "graded": True
        },
        "ResolutionEffort": {
            "type": "ResolutionEffort",
            "question": "Did the representative show effort in solving the participant's issue with the goal of achieving a solution on the call?  If yes, this is a positive answer. If no, this is a negative answer.",
            "graded": True
        },
        "DeEscalationEffort": {
            "type": "DeEscalationEffort",
            "question": "Did a situation arise during the call that required the representative to de-escalate tension or agitation from the participant? If yes, did the representative employ strategies like using empathy statements, maintaining a calm tone, and demonstrating a clear understanding of the problem to successfully de-escalate the situation? If and only if there was a palpable tension or agitation from the participant that the representative successfully de-escalated, this is a positive answer. If the call was generally pleasant, with no instances of tension or agitation that required de-escalation, this is also a positive answer. If there was tension or agitation from the participant that the representative failed to de-escalate, this is a negative answer. Explain your reasoning and reference any utterances from the participant that indicate frustration or tension.",
            "graded": True
        },
        "StrengthAssessment": {
            "type": "StrengthAssessment",
            "question": "Identify three elements of the call where the rep was performing positively. Act as a coach giving encouraging feedback to the rep for what they did right on this call. Keep it to three main points of feedback so the reader isn’t overwhelmed.",
            "graded": False,
        },
        "WeaknessAssessment": {
            "type": "WeaknessAssessment",
            "question": "Identify three elements of the call where the rep was performing poorly. Act as a coach giving feedback on what the rep can be doing better on the call to follow guidelines above. Summarize any aspects of the required process that they missed, and how they can improve next time. Keep this response concise, as to not overwhelm the reader.",
            "graded": False,
        },
        "Gist": {
            "type": "Gist",
            "question": "Generate a brief summary of this call formatted with bullet points. Use the following criteria in your response: 1. Customer's concern, 2. Whether or not the issue was resolved, and how it was resolved, 3. Any action items that are required as a result of the call. 4. Customer's sentiment throughout the call. What was it at the beginning and what was it at the end.",
            "graded": False,
        }
    }

    results = {}
    for question in promptList:

        if promptList[question]['graded']:
            response = with_function(promptList[question]['question'], system_prompt, transcript)
        elif not promptList[question]['graded']:
            response = without_function(promptList[question]['question'], system_prompt, transcript)

        try:
            if promptList[question]['graded']:
                results[question] = {
                    "type": promptList[question]['type'],
                    "graded": promptList[question]['graded'],
                    "raw": response['choices'][0]['message']['function_call']['arguments'],
                    # "grade": answer['answer'],
                    # "reasoning": answer['reasoning']
                }
            elif not promptList[question]['graded']:
                results[question] = {
                    "type": promptList[question]['type'],
                    "graded": promptList[question]['graded'],
                    "raw": response['choices'][0]['message']['content']
                }
        except Exception as e:
            print("Exception in formatting response:", json.dumps(e))
        if results[question]['type'] != "StrengthAssessment" and results[question]['type'] != "WeaknessAssessment" and results[question]['type'] != "Gist":
            results[question]['raw'] = replace_digits_with_x(results[question]['raw'])
        # print('\n'+json.dumps(results[question]))
    # returns a dictionary of results and a boolean indicating success
    return results, True

def notify_server(url, bucket, key):
    data = json.dumps({
                "bucket": bucket,
                "key": key,
                "result": '',
                "transcript": '',
                "success": '',
                "status": "Analyzing"
            })
    request = requests.post(url, data=data)
    print("Analyzing:", request)
    # print(json.dumps(request))

def lambda_handler(event, context):

    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    key_lower = key.lower()

    # -----------------------------------------------------
    # destination server info:
    # change this if you're working in different environments

    prod_domain = 'app.lightbulb.ai'
    dev_domain = 'dev.lightbulb.ai'
    api_route = '/api/call-data'

    destination_bucket = 'lightbulb-prod-output'
    # destination_bucket = 'lightbulb-dev-output'

    url = "https://"+prod_domain+api_route
    # url = "https://"+dev_domain+api_route
    # url = "http://b581-2605-a601-a0c8-dc00-6be7-5945-4f80-f17a.ngrok.io"+api_route
    # -----------------------------------------------------

    haveTranscript = False
    haveResponse = False
    openai.api_type = "azure"
    result = 'empty'
    transcript = ''
    masked_transcript = ''
    error = ''
    audio_duration = 0

    # Download file from S3
    local_file_path = '/tmp/'+key
    s3client = boto3.client('s3')
    s3client.download_file(bucket, key, local_file_path)
    print("Downloading", key, "from", bucket, local_file_path)
    openai_credentials, whisper_credentials = get_openai_credentials()

    try:

        if not key_lower.endswith(('.wav', '.mp3')) or re.search(r'\.(wav|mp3)\s*\(\d+\)$', key_lower):
            raise Exception("Bad file format")
        print("Filepath: ", local_file_path)
        audio_duration = 0
        # audio_duration = get_audio_duration(local_file_path)
        # print("Audio Duration: ", audio_duration)


        ### initialize whisper azure creds
        api_key, api_base, api_version = whisper_credentials['api_key'], whisper_credentials['api_base'], whisper_credentials['api_version']
        openai.api_key = api_key
        openai.api_base = api_base
        openai.api_version = api_version
        try:
            notify_server(url, bucket, key)
        except Exception as e:
            print("Exception while phoning home:", json.dumps(e))

        delay = 0
        maxAttempts = 5
        attempt = 0
        while (len(transcript) == 0 and attempt < maxAttempts):
            print("Beginning transcript loop")
            errorPresent = False
            try:
                # attempts to transcribe audio file
                print("Attempt #", attempt, "of", maxAttempts,"; Delay:", delay)
                time.sleep(delay)
                with open(local_file_path, 'rb') as file:
                    transcript = openai.Audio.transcribe(
                        deployment_id="whisper-model",
                        model="whisper-1",
                        file=file
                        )['text']
                print("Closing file...")
                file.close()
                    # replaces SSNs with XXX-XX-XXXX
                if len(json.dumps(transcript)) > 0:
                    print("Success generating transcript; exiting loop")
                    haveTranscript = True
            except Exception as e:
                error = e
                attempt += 1
                if delay == 0:
                    delay = 10
                else:
                    delay = delay * 2
                print("Exception in transcription:", e)
                print("Current delay:", delay, "; Attempt", attempt, "of", maxAttempts," for file ", key)


    except Exception as e:
        error = e
        print("Could not open file:", e)

    try:
        # delete the file from the /tmp directory
        os.remove(local_file_path)
        print("Deleted file /tmp/"+key)
    except Exception as e:
        error = e
        print("Exception in deleting file:", e)

    print("Transcript:", json.dumps(transcript))

    try:
        # analyze the transcript with the given prompts
        if transcript:
            result, haveResponse = analyze(transcript, openai_credentials)
    except Exception as e:
        error = e
        print("Exception in Q&A:", json.dumps(e))

    try:
        # Indicates success or failure - can be redirected to a different server
        if haveTranscript & haveResponse:
            # replaces SSNs with XXX-XX-XXXX
            masked_transcript = replace_digits_with_x(transcript)
            print("Masked transcript:",json.dumps(masked_transcript))

            destination_key = key
            move_file(bucket, key, destination_bucket, destination_key)
            data = json.dumps({
                "bucket": bucket,
                "key": key,
                "result": json.dumps(result),
                "transcript": json.dumps(masked_transcript),
                "raw_transcript": json.dumps(transcript),
                "success": "true",
                "status": "Completed"
            })
            request = requests.post(url, data=data)
            print("Success:", request)

        else:
            data = json.dumps({
                "bucket": bucket,
                "key": key,
                "result": json.dumps(result),
                "transcript": json.dumps(masked_transcript),
                "raw_transcript": json.dumps(transcript),
                "success": "false",
                "status": "Error",
                # "errorMessage": json.dumps(error)
            })
            request = requests.post(url, data=data)
            print("Error:", request)
    except Exception as e:
        data = json.dumps({
            "bucket": bucket,
            "key": key,
            "result": json.dumps(result),
            "transcript": json.dumps(masked_transcript),
            "raw_transcript": json.dumps(transcript),
            "success": "false",
            "status": "Error",
            # "errorMessage": json.dumps(error)
        })
        request = requests.post(url, data=data)
        print("Error:", request)
        print("Exception while phoning home:", json.dumps(e))

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from bucket %s-%s'%(bucket,key))
    }