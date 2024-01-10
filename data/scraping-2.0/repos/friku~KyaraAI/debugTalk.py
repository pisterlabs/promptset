import json
import os
import random
import re
import time
from pathlib import Path

import openai

from characterAI import CharacterAI, ChatController, OpenAILLM, contextDB_json
from utils.conversationRule import ConversationTimingChecker
from utils.FIFOPlayer import FIFOPlayer
from utils.getYoutubeChat import YoutubeChat
from utils.playWaveManager import WavQueuePlayer
from utils.tachieViewer import TachieViewer
from utils.textEdit import remove_chars
from utils.voicevoxUtils import makeWaveFile, text2stream, voicevoxHelthCheck
from utils.wavePlayerWithVolume import WavPlayerWithVolume


def main():
    if not voicevoxHelthCheck():
        return
    try:
        yt_url = 'https://www.youtube.com/watch?v=t6Ag3cHeCZ4' # youtubeのURL
        youtubeChat = YoutubeChat(yt_url)
        
        # キャラクターがwavファイルを作成する
        # キャラクターのwavファイルを順番に再生する
        # 立ち絵を動かす
        fifoPlayer = FIFOPlayer()
        fifoPlayer.playWithFIFO()

        # imagesDirPath = Path('characterConfig/test/images')
        # tachieViewer = TachieViewer(imagesDirPath)
        # tachieViewer.play()

        LLM = OpenAILLM()
        ruminesAI = CharacterAI(LLM, "ルミネス", contextDB_json,
                               13, 1.2, fifoPlayer, TachieViewer, 'rumines')
        rianAI = CharacterAI(LLM, "リアン", contextDB_json,
                              6, 1.2, fifoPlayer, TachieViewer, 'rian')
        ranAI = CharacterAI(LLM, "ラン", contextDB_json,
                               8, 1.2, fifoPlayer, TachieViewer, 'ran')
        neonAI = CharacterAI(
            LLM, "ネオン", contextDB_json, 7, 1.2, fifoPlayer, TachieViewer, 'neon')

        characterAIs = [ruminesAI, rianAI, ranAI, neonAI]

        conversationTimingChecker = ConversationTimingChecker()
        chatController = ChatController(characterAIs)
        chatController.initContextAll()
        chatController.addContextAll(
            'system', "[場面説明]\nあなたたちはyoutubeの配信者で,今デビュー配信中です。")

        for i in range(40):
            print("i: ", i)
            try:
                conversationTimingChecker.check_conversation_timing_with_delay(
                    fifoPlayer.get_file_queue_length)
                
                latestChat = youtubeChat.getSelectedChat()
                if latestChat is not None:
                    latestMsg = latestChat["msg"]
                    print(latestMsg)
                    for i in range(10):
                        try:
                            chatController.addComment( 'system', f"{latestMsg}")
                            break
                        except:
                            time.sleep(1)
                
                chatController.getNextCharacterResponse()


                if i % 8 == 0:
                    chatController.addContextAll(
                        'system', "[場面説明]\n別の話題を提案して、話してください。")
                elif i % 4 == 0:
                    chatController.addContextAll(
                        'system', "\n話の深堀をして話してください。")


            except openai.error.RateLimitError:
                print("rate limit error")
                time.sleep(5)
            except openai.error.APIError:
                print("API error")
                time.sleep(5)
    
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
