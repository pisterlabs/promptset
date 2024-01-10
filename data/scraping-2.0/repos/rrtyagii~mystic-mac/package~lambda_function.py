import os, json, hashlib, hmac
import openai
import requests

def webhook_verification(event, context):
    query_params = event.get('queryStringParameters', {})
    #print(query_params)
    if len(query_params) > 0:
        if query_params['hub.mode'] == 'subscribe':
            if query_params['hub.verify_token'] == 'hFKOmwZI2IcWnQGRBEjthixQYgoZ?t5qlxrv/JWtVqb5GNbY1wyboxuRsu-mQ3S7ziNoQYRHZWfrLVgJjF4aUdfwWd2!LLuKR?KqStMXF7uRSeJ9NY5Z50s7IVT45ycJ1tD2=l2UOLUgF6EyliEw?-Nir59wrIRMO6zvlq7=xWqn4zMpZlhZQIN2zr3THO6l5-ZxguowLvyXtVGCUvRQh!1xKmCr/lDPO4TQMpG502bd8hgpgOi9W9aQ1k8QwsKo':
                response = {
                    'statusCode': 200,
                    'headers': {'Content-Type': 'text/plain'},
                    'body': query_params['hub.challenge']
                }
                return response
            else:
                response= {
                    'statusCode': 403,
                    'headers': {'Content-Type': 'text/plain'},
                    'body': 'Error, wrong validation token!'
                }
                return response
        else:
            response= {
                'statusCode': 403,
                'headers': {'Content-Type': 'text/plain'},
                'body': 'Error, wrong mode!'
            }
            return response
    else:
        response= {
            'statusCode': 403,
            'headers': {'Content-Type': 'text/plain'},
            'body': 'Error, no query parameters'
        }
        return response

        
def payload_verification(event, context):
    response = {
        'statusCode': 500,
        'headers': {'Content-Type': 'text/plain'},
        'body': "something went wrong"
    }
    
    headers = event.get('headers', {})
    body = event.get('body', '')
    signature = headers['X-Hub-Signature-256']

    if(not signature):
        response['body'] = f"couldn't find {signature} in headers"
        return response
    else:
        try:
            elements = signature.split('=')
            #print("element is")
            #print(elements)

            signature_hash = elements[1]
            app_secret = '4d31132f851176903d5c39a927fdf89c'

            key = bytes(app_secret, 'UTF-8')
            expected_hash = hmac.new(key, msg=bytes(body, 'UTF-8'), digestmod=hashlib.sha256).hexdigest()

            if (signature_hash!= expected_hash):
               response["body"] = "eh " + expected_hash + " sh " + signature_hash
               #print(response)
               return response
            else:
                response["statusCode"] = 200
                response["body"] = "OK HTTPS"
                #print(response)
                return response
        except Exception as err:
            response['body'] = f"Unexpected error: {err}, type(err) = {type(err)}"
            return response


def send_message(phone_number, message, phone_number_id):
    permanent_token="EAAHjQR01zusBAOtY9p9UON9pl6tE3FwetFNa1nJ1rJ446zeBGGmaO6syKGPQdvTAGMUNzqRln2eSfkoBZBdfDdsISvF2qUzYuOkbf93zGTO4xOQwBcwjZC33B9hivtvgIwcGeOBFCo54G9CHFQz2u4zkY6LPygeSy1IGgQnmbhbjGuLsdtmeJVEy3udNbvOzuXnpXv5wZDZD"
#100739036314999
    url =f"https://graph.facebook.com/v16.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {permanent_token}",
        "Content-Type": "application/json"
    }
    data = {
      "messaging_product": "whatsapp",
      "recipient_type": "individual",
      "to": phone_number,
      "type": "text",
      "text": { 
        "preview_url": False,
        "body": message
        }
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code==200:
            #print(response.json())
            return response.json()
        else:
            #print(response.json())
            return response.json()
    except Exception as err:
        print(f"Unexpected error: {err}, type(err) = {type(err)}")


def prompt_engineering(prompt):
    openai.organization = "org-f4qxMHyg8hVnlX41W8WHv7JH"
    #openai.api_key = "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"
    try:
        ai_response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [{
                "role":"user", "content": f"{prompt}"
            }]
        )
        print(ai_response)
        return ai_response['choices'][0]['message']['content']
    except Exception as err:
        print(err)
        return err


def read_reciepts(phone_number_id, message_ID):

    permanent_token="EAAHjQR01zusBAOtY9p9UON9pl6tE3FwetFNa1nJ1rJ446zeBGGmaO6syKGPQdvTAGMUNzqRln2eSfkoBZBdfDdsISvF2qUzYuOkbf93zGTO4xOQwBcwjZC33B9hivtvgIwcGeOBFCo54G9CHFQz2u4zkY6LPygeSy1IGgQnmbhbjGuLsdtmeJVEy3udNbvOzuXnpXv5wZDZD"

    try:
        message_object = {
            'messaging_product': 'whatsapp',
            'status': 'read',
            'message_id': message_ID,
        }

        url = f"https://graph.facebook.com/v15.0/{phone_number_id}/messages"
        header = {
            "Authorization": f"Bearer {permanent_token}",
            "Content-Type":"application/json"
        }
        
        response = requests.post(url=url, headers=header, data=json.dumps(message_object))

        if response.status_code == 200:
            #print(response.json())
            return response.json()
        else:
           # print(response.json())
            return response.json()
    except Exception as e:
        print("Encountered an exception while updating read receipts")
        print(e)
        return e


def lambda_handler(event, context):
    print("event is")
    print(event)

    http_method = event.get('httpMethod', '')
    if http_method =='GET':
        webhook_verification(event, context)
    if http_method == 'POST':
        payload_result = payload_verification(event, context)
        
        if payload_result['statusCode'] == 200:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
            if body:
                entries = body['entry']
                for entry in entries:
                    for change in entry['changes']:
                        value = change['value']
                        if value:
                            phone_number_id = value['metadata']['phone_number_id']

                            if 'messages' in value:
                                if(value['messages'] is not None):
                                    for message in value['messages']:

                                        if message['type'] == "text":
                                            message_from = message['from']
                                            #print(f"message from: {message_from}")
                                            message_body = message['text']['body']
                                            #print(f"message body: {message_body}")

                                            ai_response = prompt_engineering(message_body)
                                            #print("ai_response")
                                            #print(ai_response)

                                            response = send_message(message_from, ai_response, phone_number_id)
                                            #print("send_message_response")
                                            #print(response)

                            # elif 'statuses' in value:
                            #     if(value['statuses'] is not None):
                            #         for status in value['statuses']:
                            #             if status['status'] == 'read' or status['status'] == 'sent':
                            #                 response = {
                            #                     'statusCode': 200,
                            #                     'headers': {'Content-Type': 'text/plain'},
                            #                     'body': status
                            #                 }
                            return response
        else:
            return {
                'statusCode': 403,
                'headers': {'Content-Type': 'text/plain'},
                'body': "Could not verify payload. Try again!"
            }