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
from utils.playWaveManager import WavQueuePlayer
from utils.tachieViewer import TachieViewer
from utils.textEdit import remove_chars
from utils.voicevoxUtils import makeWaveFile, text2stream, voicevoxHelthCheck
from utils.wavePlayerWithVolume import WavPlayerWithVolume
from utils.YTComment import YTComment


def main():
    if not voicevoxHelthCheck():
        return
    try:
        yt_url = 'https://www.youtube.com/watch?v=vdK6kuqjn10' # youtubeのURL
        youtubeChat = YTComment(yt_url)
        
        fifoPlayer = FIFOPlayer()
        fifoPlayer.playWithFIFO()


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
            'system', "[場面説明]\nあなたたちはyoutubeの配信者で,今デビュー配信中です。ボカロ曲について雑談してください。")
        
        
        chatController.addContextAll(
            'system', "[場面説明]\nこんにちわ、ルミネスです。今日はボカロ曲について雑談しましょう。")
        
        

        for i in range(10000):
            print("i: ", i)
            try:
                conversationTimingChecker.check_conversation_timing_with_delay(
                    fifoPlayer.get_file_queue_length)
                
                # while True:
                #     latestChat = youtubeChat.get_comment()
                #     print(latestChat)
                    
                #     if latestChat is not None:
                #         break
                #     time.sleep(1)

                latestChat = youtubeChat.get_comment()
                if latestChat is not None:
                    latestMsg = latestChat["message"]
                    chatController.addComment( 'system', f"{latestMsg}")

                
                chatController.getNextCharacterResponse()
                
                
                if i % 12 == 0:
                    chatController.addContextAll(
                        'system', "[場面説明]\n別の話題を提案して、話してください。")
                elif i % 6 == 0:
                    chatController.addContextAll(
                        'system', "[場面説明]\n話の深堀をして話してください。")



            except openai.error.RateLimitError:
                print("rate limit error")
                time.sleep(5)
            except openai.error.APIError:
                print("API error")
                time.sleep(5)
        print("end")
    
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
