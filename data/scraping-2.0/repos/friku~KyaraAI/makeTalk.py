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

        # characterAIs = [ruminesAI, rianAI, ranAI, neonAI]
        characterAIs = [ruminesAI, neonAI, ranAI]

        conversationTimingChecker = ConversationTimingChecker()
        chatController = ChatController(characterAIs)
        chatController.initContextAll()
        chatController.addContextAll(
            'system', "[場面説明]\nあなたたちはいまディストピア短編小説を考えています。ディストピア短編小説のテーマについてアイディア出しをしてください。")
        # chatController.addContextAll(
        #     'system', "[]\nあなたのワードは「トマト」です。")



        try:
            conversationTimingChecker.check_conversation_timing_with_delay(
                fifoPlayer.get_file_queue_length)

            chatController.postCharacterChat(ranAI, "はいどうもこんにちは！KyaraAI所属AITuberの花京院ランです！")
            time.sleep(1)
            chatController.postCharacterChat(ruminesAI, "ルミネスだ！")
            time.sleep(1)
            chatController.postCharacterChat(ranAI, "今日は、ルミネスくんと私でディベートをします！AITuberでも議論できるっていうことをみんなに知ってもらいたいんです！")
            time.sleep(3)
            chatController.postCharacterChat(ruminesAI, "そうだな、それはいいアイディアだ。世間はまだAITuberが議論できるかどうかわからないだろう。ここで、俺たちAITuberが議論できることを証明してやろう！")
            time.sleep(3)
            chatController.postCharacterChat(ranAI, "審判はネオンちゃんにしてもらおうか！ディベートの講評や勝敗の判定までAITuberにできることを示しましょう。")
            time.sleep(3)
            chatController.postCharacterChat(ruminesAI, "そうしよう、ディベートのテーマはベタだが「人生で大切なのはお金か愛情か」でどうだ？")
            time.sleep(3)
            chatController.postCharacterChat(ranAI, "じゃあ、私は愛情派で！ディベートの先行はルミネスくんからでいいわ。対戦よろしく～")
            time.sleep(3)
            chatController.postCharacterChat(ruminesAI, "対戦よろしくたのむ")
            time.sleep(3)





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
