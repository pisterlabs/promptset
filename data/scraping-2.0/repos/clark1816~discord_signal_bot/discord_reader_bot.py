import requests
import json
import datetime
import pytz
import openai
import time
import re
import config

openai.api_key = config.key
messages = [
    {"role": "system", "content": "Please read through these Discord messages and tell me the 'stock_ticker', 'strike_price', and 'call_or_put' in JSON format output. cal or put shoud be either 'call' or 'put'"},
]

wait_time = 10 # second
prev_msg_id = 0

start_time = datetime.datetime.now().astimezone(pytz.UTC).timestamp()

def retrieve_messages(channelid):
    global prev_msg_id
    global start_time
    new_msg = True

    headers = {
        'authorization': config.auth_key
    }
    r = requests.get(f'https://discord.com/api/v9/channels/{channelid}/messages', headers=headers)
    jsonn = json.loads(r.text)

    if prev_msg_id == jsonn[0]['id']:
        new_msg = False

    all_message = []
    for value in jsonn:
        timestamp_str = value['timestamp']
        dt = datetime.datetime.fromisoformat(value['timestamp'])
        timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()

        if timestamp > start_time:
            dt = datetime.datetime.fromtimestamp(timestamp).astimezone(pytz.UTC)
            dt_str = dt.strftime('%H:%M:%S')
            content = value['content']
            username = value['author']['username']

            message_line = {"username": username, "time": dt_str, "message": content}
            all_message.append(message_line)

    prev_msg_id = jsonn[0]['id']
    start_time = datetime.datetime.now().astimezone(pytz.UTC).timestamp()

    all_message.reverse()

    if len(all_message) == 0:
        new_msg = False

    return json.dumps(all_message), new_msg


while True:
    all_message, new_msg = retrieve_messages(config.channel_id)  # Replace CHANNEL_ID with the actual channel ID

    if not new_msg:
        print(f"No new message, waiting for {wait_time} sec.")
        time.sleep(wait_time)
        continue
    else:
        print(f"Running chat-gpt ...")

    if all_message:
        messages.append({"role": "user", "content": all_message})
        response = openai.ChatCompletion.create( model="gpt-4", messages=messages)
        messages = messages[:1]  # Clear all messages except the system message
        system_response = response["choices"][0]["message"]["content"]

        print(system_response)

        try:
            output_json = json.loads(system_response)
            json_keys = output_json[0].keys()

            if "stock_ticker" in json_keys and "strike_price" in json_keys and "call_or_put" in json_keys:
                for data in output_json:
                    try:
                        stock_ticker = data["stock_ticker"]
                        strike_price = float(data["strike_price"])
                        call_or_put = data["call_or_put"]

                        assert stock_ticker!="", "Stock ticker is empty"
                        assert call_or_put in ["call", "put"], "call or put is not in correct form"
                        print("Enter trade", stock_ticker, strike_price, call_or_put)
                        enter_trade(stock_ticker, strike_price, call_or_put)
                    except AssertionError as error:
                        print("Assertion Error:", error)
                    except Exception as e:
                        print("Something missing please check: ", data)
                        print("Error: ", e)
                    
        except:
            print("Chat-gpt output is not in json format")
            print(system_response)
            prev_msg_id = 0 # this will allow to re-run chatgpt with same input again. Expecting to get correct output this time. 
       

    print(f'waiting for {wait_time} sec')
    time.sleep(wait_time)  # Wait for 10 seconds before retrieving new messages


