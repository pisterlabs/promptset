import azure.cognitiveservices.speech as speechsdk
import time
import openai
import os
from websocket_server import WebsocketServer
import threading
import signal
import sys
import json

functions = [
	{
		"name":"trainFunction",
		"description": "電車の運行情報を教えてください。路線名は以下から検索してください[山手線, 上野東京ライン, 湘南新宿ライン, 相鉄線直通列車, 東海道線, 京浜東北線, 横須賀線, 南武線, 横浜線, 伊東線, 相模線, 鶴見線, 宇都宮線, 高崎線, 埼京線, 川越線, 武蔵野線, 上越線, 信越本線, 吾妻線, 烏山線, 八高線, 日光線, 両毛線, 中央線快速電車, 中央・総武各駅停車, 中央本線, 五日市線, 青梅線, 小海線, 常磐線, 常磐線快速電車, 常磐線各駅停車, 水郡線, 水戸線, 総武快速線, 総武本線, 京葉線, 内房線, 鹿島線, 久留里線, 外房線, 東金線, 成田線, 羽越本線, 羽越本線, 羽越本線, 羽越本線, 奥羽本線, 奥羽本線（山形線）, 仙山線, 仙山線, 仙石線, 仙石東北ライン, 東北本線, 磐越西線, 左沢線, 石巻線, 大船渡線, 大船渡線ＢＲＴ, 大湊線, 男鹿線, 釜石線, 北上線, 気仙沼線, 気仙沼線ＢＲＴ, 五能線, 五能線, 五能線, 田沢湖線, 只見線, 津軽線, 八戸線, 花輪線, 磐越東線, 山田線, 米坂線, 陸羽西線, 陸羽東線, 羽越本線, 羽越本線, 篠ノ井線, 白新線, 飯山線, 越後線, 大糸線, 弥彦線, 成田エクスプレス, ひたち・ときわ, あずさ・かいじ・富士回遊, はちおうじ・おうめ, しなの, わかしお・さざなみ, しおさい, 日光・きぬがわ, 草津・四万・あかぎ, 踊り子・湘南, いなほ・しらゆき, つがる, サンライズ瀬戸・出雲, 臨時列車, 東北新幹線, 山形新幹線, 秋田新幹線, 北陸新幹線, 上越新幹線, 銀座線, 丸ノ内線, 日比谷線, 東西線, 千代田線, 有楽町線, 半蔵門線, 南北線, 副都心線, 日光線特急, 伊勢崎線特急, ＳＬ・ＤＬ, ＴＨライナー, 東武スカイツリーライン〜久喜・南栗橋, 伊勢崎線 久喜 以北エリア, 日光線 南栗橋 以北エリア, 亀戸線・大師線, 東武アーバンパークライン, ＴＪライナー, 東上本線, 越生線, 浅草線, 三田線, 新宿線, 大江戸線, 京王線, 東横線, 目黒線, 東急新横浜線, 田園都市線, 大井町線, 池上線, 東急多摩川線, 世田谷線, こどもの国線]",
		"parameters":{
			"type": "object",
                "properties": {
					"lineName": {
                        "type": "string",
                        "description": "聞かれている路線"
					}
			},
			"requaired":["lineName"]
		}

	},
    {
		"name":"weatherFunction",
		"description": "天気を教えてください",
		"parameters":{
			"type": "object",
                "properties": {
			},
			"requaired":[]
		}
	},
    {
		"name":"timeTableFunction",
		"description": "時間割を教えてください",
		"parameters":{
			"type": "object",
                "properties": {
                    "dayNumber": {
                        "type": "string",
                        "description": "聞かれている曜日を表す整数 (0:指定なし, 1:月曜日, 2:火曜日, 3:水曜日, 4:木曜日, 5:金曜日, 6:土曜日)"
					}
			},
			"requaired":["dayNumber"]
		}
	},
    {
		"name":"dateTimeFunction",
		"description": "日付もしくは時間を聞かれた際に答える",
		"parameters":{
			"type": "object",
                "properties": {
			},
			"requaired":[]
		}
	},
    {
        "name":"newsFunction",
        "description": "ニュースについて聞かれた際に答える",
        "parameters" :{
            "type": "object",
            "properties":{
            },
            "requaired":[]
        }
    }
    
]

# Global Variables
server = None
session_active = False
messages_history = []
last_input_time = time.time()
recognized_text = ""
clients = {}

# SIGINTハンドラ関数
def signal_handler(sig, frame):
    global server
    print("Shutting down gracefully...")
    server.shutdown()
    sys.exit(0)

# WebSocketサーバーのコールバック関数
def new_client(client, server):
    print("New client connected")

def client_left(client, server):
    print("Client disconnected")

def message_received(client, server, sned_message):
    print("Client said: " + sned_message)
    payload = json.loads(sned_message)
    if payload['type'] == 'CONNECT':
        clients[payload['name']] = client

    if payload['type'] == 'RESPONSE':
        contents_data = str(payload['data'])
        
        if(client == clients['MMM-trainInfo']):
            messages_history.append({"role": "user", "content": "以下のデータを使って一つ前の質問に答えてください" + contents_data})
            
        if(client == clients['weather']):
            messages_history.append({"role": "user", "content": "以下のフォーマットとデータを使って一つ前の質問に答えてください。#フォーマット[現在の天気は~,気温は~です] #データ" + contents_data})

        if(client == clients['MMM-timeTable']):
            if(json.loads(contents_data) == []):
                speak_and_dispaly("時間割はありません。")
                return
            messages_history.append({"role": "user", "content": "以下のデータを読み上げてください。授業がない場合はスキップしてください。" + contents_data})

        if(client == clients['clock']):
            messages_history.append({"role": "user", "content": "以下のデータを使って一つ前の質問に日本語で答えてください" + contents_data})

        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=messages_history,
        )
        speak_and_dispaly(str(response['choices'][0]['message']['content']))
        global session_active
        session_active = False

def get_openai_response(text):
    global messages_history

    try:
        # ユーザーメッセージを履歴に追加
        messages_history.append({"role": "user", "content": text})

        # OpenAIのレスポンスを取得
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=messages_history,
            functions=functions,
            function_call="auto"
        )
        response_data = response['choices'][0]['message']

        # 機能呼び出しの処理
        if 'function_call' in response_data:
            function_call = response_data['function_call']
            function_name = function_call['name']
            arguments = json.loads(function_call['arguments'])

            if function_name == 'trainFunction':
                lineName = arguments['lineName']
                server.send_message(clients['MMM-trainInfo'], f'{{"type":"CALL","lineName":"{lineName}"}}')
            elif function_name == 'weatherFunction':
                server.send_message(clients['weather'], '{"type":"CALL"}')
            elif function_name == 'timeTableFunction':
                dayNumber = arguments.get('dayNumber', time.localtime().tm_wday + 1)
                server.send_message(clients['MMM-timeTable'], f'{{"type":"CALL","dayNumber":{dayNumber}}}')
            elif function_name == 'dateTimeFunction':
                server.send_message(clients['clock'], '{"type":"CALL"}')
            return ""

        # 通常のレスポンスの処理
        else:
            messages_history.append({"role": "assistant", "content": response_data['content']})
            return response_data['content']


    except Exception as e:
        print("Exception:", e)
        return "すみません、よくわかりませんでした."

def setup_websocket_server():
    global server
    server = WebsocketServer(host="127.0.0.1", port=5005)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    thread = threading.Thread(target=server.run_forever)
    thread.start()

def recognized(evt):
    global recognized_text
    if evt.result.text == "":
        return
    print('「{}」'.format(evt.result.text))
    recognized_text = evt.result.text

def check_activation_phrase(text):
    activation_phrases = ["鏡よ", "鏡を","スマートミラー"]
    return any(phrase in text for phrase in activation_phrases)

def handle_activation():
    global session_active, messages_history
    session_active = True
    messages_history = [{"role": "system", "content": "あなたは与えられた情報を使って質問に答えることができます。質問に答える際は質問に対する回答をなるべく簡潔に答えてください。"}]
    speak_and_dispaly("はい、なんでしょう？")

def speak_and_dispaly(text):
    print(text)
    speech_recognizer.stop_continuous_recognition()

    #clientsにMMM-chatがあるか確認
    if('MMM-chat' in clients):
        server.send_message(clients['MMM-chat'],'{"type":"TEXT","text":"' + text.replace('\n','') + '"}')
    speech_synthesizer.speak_text(text)
    speech_recognizer.start_continuous_recognition()


def main_loop():
    global recognized_text, last_input_time, session_active, messages_history

    while True:
        time.sleep(1)
        speech_recognizer.stop_continuous_recognition()
        recognized_text = input()
        if recognized_text == "":
            continue
        
        if session_active:
        #if True:
            messages_history = []
            speech_recognizer.stop_continuous_recognition()
            response_text = get_openai_response(recognized_text)
            if response_text:
                session_active = False
                speak_and_dispaly(response_text)

        elif check_activation_phrase(recognized_text):
            handle_activation()
        recognized_text = ""

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    setup_websocket_server()

    # Azure and OpenAI setup
    openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    openai.api_version = "2023-07-01-preview"
    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")

    speech_key = os.getenv('AZURE_SPEECH_KEY')
    service_region, language = "japaneast", "ja-JP"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region, speech_recognition_language=language)
    audio_input_config = speechsdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input_config)

    audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name = 'ja-JP-NanamiNeural'
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

    print("話しかけてください...")

    speech_recognizer.recognized.connect(recognized)
    speech_recognizer.start_continuous_recognition()

    main_loop()
