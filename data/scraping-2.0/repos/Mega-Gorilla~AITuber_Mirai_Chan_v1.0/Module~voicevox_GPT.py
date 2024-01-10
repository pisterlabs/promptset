
import os
import openai
import time
import re
import json
import websocket
import Module.VtubeStudio_API as VtubeStudio_API
import asyncio
from Module.voicevox import Add_preset,update_preset, audio_stream_start, text_to_wave, audio_stream_stop , Get_presets, remove_all_presets
from pprint import pprint
from Module.print_color import print_color

def VoiceVoxGPT_AutoPich(OpenAI_key,emotion,AI_model = 'gpt-3.5-turbo',max_retries=3):
    """
    この関数は、OpenAIのGPT-3.5モデルを利用して、指定した感情に基づくLive2Dパラメータを生成します。
    生成したパラメータはVtubeStudio_emotionに提供することで、Live2Dモデルの動きに変換することができます。
    この関数の応答には10秒から30秒ほど時間がかかります。キャラクターをアニメーションさせる場合は、先にすべてのモーションを保存しておくことをお勧めします。

    引数:
    OpenAI_key -- OpenAIのAPIキー [String]
    emotion -- 表現したい感情 [String]
    AI_model -- OpenAI社が提供しているモデルを設定 [String]

    戻り値:
    dictionary -- VtubeStudio_APIに渡すためのLive2Dパラメータの辞書を返します。Live2Dのパラメータ辞書はVtubeStudio_emotionのemotion_listに用いることができます。取得出来なかった場合はFalseを返します。 [Dictionary]
    """
    openai.api_key=OpenAI_key
    Prompt_template = """I would like you to estimate the vocal quality of a character's speech based on their emotions.
## Rules
Estimate the character's 'speakingSpeed','pitch', and 'intonation' from their emotions.
The 'speakingSpeed' should be selected from 'fast, 'medium', 'slow',
The 'pitch' should be selected from 'high', 'normal', 'low',
The 'intonation' should be selected from 'high', 'normal', 'low'.
These values need to be adjusted based on the content and description.

##Example
Input: excited about Mario Kart and teasingly criticizes someone's bad skills.
Output: {"speakingSpeed":"fast","pitch":"high","intonation":"high"}"""

    Prompt = Prompt_template
    result = ""
    #print_color(f"Prompt:{Prompt}","green")

    #GPT3.5にて問い合わせ
    #print_color(f'{AI_model} Stream ViceVox問い合わせ開始 ','yellow',True)
    GPT_time= time.time()

    #システムプロンプトの追加
    ChatGPT_Prompt=[{"role": "system", "content": Prompt}]
    #ユーザープロンプトの追加
    ChatGPT_Prompt.append({"role": "user", "content": emotion})
    for i in range(max_retries):
        try:
            for chunk in openai.ChatCompletion.create(
                model=AI_model,
                messages=ChatGPT_Prompt,
                stream=True,
            ):
                content = chunk["choices"][0].get("delta", {}).get("content")
                #print_color(f"{content}",'green',True)
                if content is not None:
                    if '}' in content:
                        result +=content
                        break
                    result += content
            # 終了したらループを抜ける
            break
        except openai.error.RateLimitError:
            # エラーメッセージを表示
            print_color(f"モデルが他のリクエストで過負荷となっています。")
            # エラーが持続する場合はOpenAIのヘルプセンターに連絡を促すメッセージを表示
            print_color(f"エラーが持続する場合は、help.openai.com に連絡してください。")
            # リトライの間隔を設定（ここでは例として1秒）
            time.sleep(1)
        except Exception as e:
            # その他のエラーを表示
            print_color(f"OpenAI問い合わせ中に、予期せぬエラーが発生しました：{e}")
            break
    
    #GPTの結果を取得
    print_color(f"[自動声質]{AI_model} 問い合わせ時間: {(time.time()-GPT_time):.1f}sec","yellow")
    #print(result)
    #Json部分のみを切り取りして、辞書配列に変換
    match = re.search(r'\{(.|\n)+\}', result)
    if match is not None:
        json_string = match.group()
        try:
            dictionary = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"[自動声質] JSON decode error: {e}\nResult: {result}")
            # ここでエラー処理を行うか、単にパスして次の処理に進みます。
            pass
    else:
        # 一致が見つからなかった場合のエラー処理を行う
        print_color("エラー: モーションについて問い合わせを行ったがモーションが正しく取得できなかった。")
        print_color(f"{result}")
        return False
    return dictionary

def Create_AutoPich_Preset(characterPreset,voice_data,presetid=0):
    '''Example characterPreset = {
        'CharacterName': 'みらい',
        'speaker_uuid': '882a636f-3bac-431a-966d-c5e6bba9f949',
        'style_id':47,
        'speakingSpeed':{'min':0.85,'default':1.0,'max':1.2},
        'pitch':{'min':-0.05,'default':0.0,'max':0.04},
        'intonation':{'min':0.6,'default':1.0,'max':1.4}
        }

        voice_data = {"speakingSpeed":"fast","pitch":"high","intonation":"high"}'''
    
    for key,value in voice_data.items():
        if value == 'fast' or value == 'high':
            voice_data[key] = characterPreset[key]["max"]
        elif value == "medium" or value == 'normal':
            voice_data[key] = characterPreset[key]["default"]
        elif value == "low" or value == 'slow':
            voice_data[key] = characterPreset[key]["min"]
        else:
            voice_data[key] = characterPreset[key]["default"]
    
    #Check Preset
    allpresets=Get_presets()
    #設定プリセットがある場合
    if any(d['id'] == presetid for d in allpresets):
        update_preset(presetid,'mirai_preset',speaker_uuid=characterPreset['speaker_uuid'], style_id=characterPreset['style_id'],speedScale=voice_data['speakingSpeed'],pitchScale=voice_data['pitch'],intonationScale=voice_data['intonation'])
    else:
        Add_preset(presetid,'mirai_preset',speaker_uuid=characterPreset['speaker_uuid'], style_id=characterPreset['style_id'],speedScale=voice_data['speakingSpeed'],pitchScale=voice_data['pitch'],intonationScale=voice_data['intonation'])

if __name__ == "__main__":
    
    # OpenAIのAPIキーを設定します
    OpenAI_key = os.getenv("OPENAI_API_KEY")
    emotion = 'Excited amusement for Mario Kart, playful teasing, and positive encouragement.'
    characterPreset = {
        'CharacterName': 'みらい',
        'speaker_uuid': '882a636f-3bac-431a-966d-c5e6bba9f949',
        'style_id':47,
        'speakingSpeed':{'min':0.85,'default':1.0,'max':1.2},
        'pitch':{'min':-0.05,'default':0.0,'max':0.04},
        'intonation':{'min':0.6,'default':1.0,'max':1.4}
        }
    #result = asyncio.run( VoiceVoxGPT_AutoPich(OpenAI_key,emotion))
    result = VoiceVoxGPT_AutoPich(OpenAI_key,emotion)
    print_color(f'{result}')
    Create_AutoPich_Preset(characterPreset,result)
    text_to_wave("音声テスト",0,speaker=47)