import openai, requests, time
from openai import OpenAI

# image detection and filter
def image_processing(url, key):
    try:
        client = OpenAI(api_key=key)
        # flags = ("1", "I'm sorry, I can't assist with that request.", openai.BadRequestError)
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a harmful content detector, if a image sent to you depicts harmful or inappropriate content beyond resonable doubt, respond with a 1, otherwise, respond with a 2. Under no circumstances should you respond with anything other than a 1 or a 2.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        print(response.choices[0].message.content)
        if response.choices[0].message.content != "1":
            print("image good")
            return False
        print("image bad")
        return True

    except Exception:
        return True

# file filter
def file_processing(file_content, key):
    url = 'https://www.virustotal.com/vtapi/v2/file/scan'

    params = {'apikey': key}

    # files = {'file': ('myfile.exe', open('myfile.exe', 'rb'))}
    data = {'file': file_content}
    response = requests.post(url, params=params, data=data)  

    time.sleep(5) # slow post can cause a race condition

    resource = response.json().get("resource")
    # print(response.text)

    url = 'https://www.virustotal.com/vtapi/v2/file/report'

    params = {'apikey': key, 'resource': resource}

    response = requests.get(url, params=params)
    positives = response.json().get("positives")
    print(positives)
    if positives == 0:
        return False
    return True


def url_processing(url, key):

    url = "https://www.virustotal.com/vtapi/v2/url/scan"

    payload = {
        "apikey": key,
        "url": url
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(url, data=payload, headers=headers)

    time.sleep(3) # slow post can cause a race condition

    resource = response.json().get("resource")
    # print(response.text)

    url = 'https://www.virustotal.com/vtapi/v2/url/report'

    params = {'apikey': key, 'resource': resource}

    response = requests.get(url, params=params)
    positives = response.json().get("positives")
    print(positives)
    if positives == 0:
        return False
    return True



