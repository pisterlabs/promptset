import openai
import time
import requests

def send_request(message):
    url = "http://192.168.1.13/print?txt="
    arduinoResponse = requests.get(url+message)
    if (arduinoResponse.status_code == 200):
        print(f"Sent to Arduino OK: {message}")
def gen(prompt):
    
    openai.api_key = "Your_API_Key"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=1000,        
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )#model 
    return response.choices[0].text

def notify(message):
    from plyer import notification
    notification.notify(
        title = 'Valid San :3',
        message = message,
        app_icon = None,
        timeout = 30,
    )


while True:
    if time.strftime('%M') == '00':
        notify(gen('Give me validation and motivation to get through my day. (give a simple paragraph of 25 words as output.'))
        send_request(gen('Give me validation and motivation to get through my day. (give a simple paragraph of 25 words as output.'))
        print('Did my job going to sleep now byeeee <3 ',time.strftime("%H:%M:%S"))
        time.sleep(50*60)
        print('I am awake now, Back to work. ',time.strftime("%H:%M:%S"))
    else:
        # #print this very minute print(time.strftime("%H:%M:%S"),": I am alive.")
        # print(time.strftime("%H:%M:%S"),": I am alive.")
        # #sleep for 10 min 
        # time.sleep(600)
        continue
