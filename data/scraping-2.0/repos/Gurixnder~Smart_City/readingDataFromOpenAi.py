import requests
from dotenv import load_dotenv
import os

class dataAI:
    def __init__(self):
        
        load_dotenv()

        myThingSpeakAPI = 'YULQKFZEJGS2J2LL'
        baseURL = 'https://api.thingspeak.com/update?api_key=%s' %myThingSpeakAPI

        OPENAI_API_KEY = "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"
        
#         def fetch():
#             
#             THINGSPEAK_CHANNEL_API_URL = 'https://api.thingspeak.com/channels/2350013/.json?api_key=K5EABMKHIXQI3IHG&results=2'
#             
#             try:
#                 response = requests.get(THINGSPEAK_CHANNEL_API_URL)
#                 response.raise_for_status()

        def fetch_weather_data():
            THINGSPEAK_CHANNEL_API_URL = "https://api.thingspeak.com/channels/2350013/feeds.json?api_key=K5EABMKHIXQI3IHG&results=2"
            response = requests.get(THINGSPEAK_CHANNEL_API_URL)
            print(response.status_code)
            if response.status_code == 200:
                feeds = response.json().get('feeds',[])
                if feeds:
                    return feeds
                else:
                    print('no data in channel')
            else:
                print("Failed to retrieve data form thingspeak")
            return None

        def analyze_data_with_openai(data):
            headers ={
                
                'Content-Type' : 'application/json',
                'Authorization' : f'Bearer {OPENAI_API_KEY}'
                
                }
            prompt = f'I have weather of room data: {data}. give me numeri value for temperature and humidity according to the values of field1 and field2'
            data = {
                    'prompt' : prompt,
                    'max_tokens' : 150
                    
                    }
           # print(data)
            
            response = requests.post('https://api.openai.com/v1/engines/text-davinci-003/completions', headers=headers, json=data)
            if response.status_code == 200:
                text = response.json().get('choices', [{}])[0].get('text', '').strip()
                return text
            else:
                print('Failed to analyze response from openai')
            return None

        def extract_numbers(data_str):
            numbers = []
            current_num = ''
            for char in data_str:
                if char.isdigit() or char=='.':
                    current_num+=char
                elif current_num != '':
                    numbers.append(float(current_num))
                    current_num=''
            if current_num != '':
                numbers.append(float(current_num))
            return numbers

        def convert_to_integers(number_list):
            return [int(num) for num in number_list]

        def rundata():
            if not OPENAI_API_KEY:
                print('yes')
            weather_data = fetch_weather_data()
#             print(weather_data)
            if weather_data:
                analysis = analyze_data_with_openai(weather_data)
                if analysis:
                    print("Data analysis: ", analysis)
                else:
                    print("Failed to analyze weather data.")
            else:
                print("No weather data to analyze.")
            temp = extract_numbers(analysis)
            temp2 = convert_to_integers(temp)
            #print(temp2[0], temp2[1])
            
            update_thingspeak(temp2[0], temp2[1])
                
        def update_thingspeak(field1_value,field2_value):
            thingspeak_url = baseURL
            data = {
                
                    'api_key':myThingSpeakAPI,
                    'field3': field1_value,
                    'field4': field2_value
                }
            response = requests.post(thingspeak_url, data=data)
            if response.status_code == 200:
                print('Updated forecast is available')
            else:
                print('Failed to send data', respose.status_code)
            
        rundata()
            
            

            
            
            
            

    
    