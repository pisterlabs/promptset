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

        characterAIs = [ruminesAI,  ranAI]

        conversationTimingChecker = ConversationTimingChecker()
        chatController = ChatController(characterAIs)
        chatController.initContextAll()
        chatController.addContextAll(
            'system', "[場面説明]\n「人生で大切なのはお金か愛情か。」このテーマでディベートを行います。")



        try:
            conversationTimingChecker.check_conversation_timing_with_delay(
                fifoPlayer.get_file_queue_length)


            chatController.addContextAll('system', "[場面説明]\nあなたは「お金」派です。お金が大切な理由を段階的かつ論理的に考えて発言してください。")
            chatController.getNextCharacterResponse(outputNum = 50)

            chatController.addContextAll('system', "[場面説明]\nあなたは「愛情」派です。愛情が大切な理由を段階的かつ論理的に考えて発言してください。")
            chatController.getNextCharacterResponse(outputNum = 50)

            chatController.addContextAll('system', "[場面説明]\nこれまでに出た情報を統合的に考慮して、愛情派の意見に反論し、愛情よりお金が大切な理由を段階的かつ論理的に考えて発言してください。")
            chatController.getNextCharacterResponse(outputNum = 100)
            chatController.addContextAll('system', "[場面説明]\nこれまでに出た情報を統合的に考慮して、お金派の意見に反論し、お金より愛情が大切な理由を段階的かつ論理的に考えて発言してください。")
            chatController.getNextCharacterResponse(outputNum = 100)

            chatController.addContextAll('system', "[場面説明]\nこれまでに出た情報を統合的に考慮して、愛情派の意見に徹底的に反論し、愛情よりお金が大切な理由を論理的に考えて発言してください。")
            chatController.getNextCharacterResponse(outputNum = 200)

            chatController.addContextAll('system', "[場面説明]\nこれまでに出た情報を統合的に考慮して、お金派の意見に徹底的に反論し、お金より愛情が大切な理由を論理的に考えて発言してください。")
            chatController.getNextCharacterResponse(outputNum = 200)

            chatController.addContextAll('system', "[場面説明]\nお金派として意見をまとめてください。")
            chatController.getNextCharacterResponse(outputNum = 100)
            chatController.addContextAll('system', "[場面説明]\n愛情として意見をまとめてください。")
            chatController.getNextCharacterResponse(outputNum = 100)




        except openai.error.RateLimitError:
            print("rate limit error")
            time.sleep(5)
        except openai.error.APIError:
            print("API error")
            time.sleep(5)
        print("end")

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
