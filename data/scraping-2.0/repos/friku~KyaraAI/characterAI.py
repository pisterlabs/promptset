'''
キャラクターAIに必要な機能の抽象を統合したクラス
統合する機能
・キャラクターの設定の読み込み
・コメントからユーザーの特定
・ユーザーごとの過去の会話の読み込み
・

'''
import json
import os
import pdb
import random
import re
import time
from pathlib import Path

import openai

from utils.chatRecorder import DictRecorder
from utils.contextDB import contextDB_json
from utils.conversationRule import ConversationTimingChecker
from utils.OpenAILLM import OpenAILLM
from utils.playWaveManager import WavQueuePlayer
from utils.PromptMaker import PromptMaker
from utils.selectCharacter import (find_string_positions,
                                   select_random_next_character)
from utils.textEdit import addText, remove_chars
from utils.TextReplacer import replace_words
from utils.voicevoxUtils import makeWaveFile, text2stream, voicevoxHelthCheck
from utils.wavePlayerWithVolume import WavPlayerWithVolume


class CharacterAI():
    def __init__(self, LLM, characterName: str, contextDBClass: contextDB_json, speakerID: int, speedScale: float = 1.0, fifoPlayer = None, TachieViewer = None, charaNameEN = 'test'):
        self.LLM = LLM
        self.characterName = characterName
        self.charaNameEN = charaNameEN
        self.characterDir = Path('./characterConfig') / characterName
        self.contextPath = self.characterDir / 'context.json'
        self.identityPath = self.characterDir / 'identity.txt'
        if not Path(f'./characterConfig/{characterName}').exists():
            assert False, f'キャラクターの設定ディレクトリが存在しません。{characterName}'
        self.contextDB = contextDBClass(self.contextPath)
        if self.contextDB.get() == []:
            self.addIdentity(self.identityPath)
        self.speakerID = speakerID
        self.speedSclae = speedScale
        self.fifoPlayer = fifoPlayer
        self.TachieViewer = TachieViewer
        self.playTachieViewer()
        self.yt_comment = ''
        self.promptMaker = PromptMaker(self.contextDB, self.identityPath, self.characterName)

    def initContext(self):
        self.contextDB.init()
        self.addIdentity(self.identityPath)

    def formatResponse(self, response):
        try:
            splitResponse = re.split("としての発言]|\n", response["content"])
            talkResponse = splitResponse[-1]
            formatResponse = f'[{self.characterName}としての発言]\n{talkResponse}'
        except:
            import pdb
            pdb.set_trace()
        return {"formatResponse": formatResponse, "talkResponse": talkResponse}

    def getResponse(self, outputNum:int = 20):
        prompt = self.promptMaker.getPrompt(outputNum=outputNum)
        start = time.time()
        response = self.LLM.getResponse(prompt)
        print(response)
        print(f"getResponse time: {time.time() - start}")
        formatResponse = self.formatResponse(response)["formatResponse"]
        print(formatResponse)
        talkResponse = self.formatResponse(response)["talkResponse"]
        self.text2VoiceObject(talkResponse)
        return {'formatResponse':formatResponse, 'talkResponse':talkResponse, 'response':response}
    
    def text2VoiceObject(self, text: str, commentFlag = False):
        cleanedTalkResponse = replace_words(text, 'dictionary.json')
        cleanedTalkResponse = remove_chars(cleanedTalkResponse, "「」 『』・") # 会話の中にある特殊文字を削除
        wavPath =  Path("./tmpWaveDir") / self.getFileName('wav')
        makeWaveFile(self.speakerID, cleanedTalkResponse,wavPath, self.speedSclae) # 音声合成
        self.setVoiceObject(wavPath, text, commentFlag) # 音声合成した音声をキューに追加
    

    def setVoiceObject(self, wavPath:Path, text = None, commentFlag = False):
        if self.fifoPlayer is None or self.tachieViewer is None:
            return
        if commentFlag: # コメントの読み上げの場合は、テキストを表示しない
            text = ''
        self.fifoPlayer.setObject(WavPlayerWithVolume(wavPath, self.tachieViewer.setMouthOpenFlag, text, self.characterName, self.yt_comment))
    
    def addContext(self, role, message):
        self.contextDB.add(role, message)

    def addIdentity(self, identityPath):
        with open(identityPath, 'r', encoding="utf-8") as f:
            identityContext = f.read()
        self.contextDB.add("system", identityContext)


    def text2speach(self, text: str):
        text2stream(self.speakerID, text)
    
    def playTachieViewer(self):
        if self.TachieViewer is None:
            return
        self.tachieViewer = self.TachieViewer(self.characterDir / 'images', self.charaNameEN)
        self.tachieViewer.play()
        
    def getFileName(self, extention: str):
        FileName = time.strftime(
            '%Y%m%d_%H%M%S', time.localtime()) + f'_({self.characterName}).{extention}'
        return FileName
        


class ChatController():
    def __init__(self, characterAIs: list):
        self.characterAIs = characterAIs
        self.speakerName = ''
        self.chatLogsDir = Path('./chatLogs')
        self.latest_yt_comment = ''
        self.attendants = [characterAI.characterName for characterAI in self.characterAIs]
        self.makeChatLog()
        self.systemContext = DictRecorder(Path('./tmpChatLog') / 'systemContext.json')
        self.characterAIsDict = {characterAI.characterName:characterAI for characterAI in self.characterAIs}

    def initContextAll(self):
        for characterAI in self.characterAIs:
            characterAI.initContext()

    def addContextAll(self, role, message, speaker=''):  # 発言を全員(自分も含める)の文脈に追加する
        for characterAI in self.characterAIs:
            characterAI.addContext(role, message)
        addText(f'{message}\n', self.chatLogsDir / self.logFileName) # chatLogに発言を追加
        self.systemContext.add({'role':role, 'content':message, 'speaker':speaker})
        
    
    def addComment(self, role, message):
        formatResponse = f'[コメント欄の文字列]\n{message}'
        self.addContextAll(role, formatResponse, speaker='コメント欄')
        self.latest_yt_comment = message
        

    def getCharacterResponse(self, characterAI, outputNum:int = 20):
        characterAI.yt_comment = self.latest_yt_comment
        if characterAI.yt_comment != '':
            characterAI.text2VoiceObject(characterAI.yt_comment, commentFlag=True)
        self.latest_yt_comment = ''
        response = characterAI.getResponse(outputNum)
        characterAI.yt_comment = ''
        self.addContextAll("user", response['formatResponse'], speaker=characterAI.characterName)
        return response
    
    def postCharacterChat(self, characterAI, message):
        characterAI.text2VoiceObject(message)
        response = f'[{characterAI.characterName}としての発言]\n{message}'
        self.addContextAll("user", response, speaker=characterAI.characterName)


    def selectSpeaker(self):
        calledSpeaker = self.getCalledSpeaker()
        if calledSpeaker is not None:
            return calledSpeaker

        pre_speakerName = self.speakerName
        nextSpeakerDict  = select_random_next_character(pre_speakerName, self.characterAIsDict)
        self.speakerName = nextSpeakerDict['name']
        print(pre_speakerName,self.speakerName)
        # pdb.set_trace()
        nextSpeaker = nextSpeakerDict['speaker']
        return nextSpeaker
        

    def getNextCharacterResponse(self, outputNum:int = 20):  # 次のキャラクターの発言を取得し文脈に追加
        speaker = self.selectSpeaker()
        response = self.getCharacterResponse(speaker, outputNum)
        return response

    def makeChatLog(self):
        #　会話内容を記録するファイルを作成
        # すべての参加キャラの名前取得し、_でつなげる
        allAttendants = '_'.join(self.attendants)
        self.logFileName = time.strftime(
            '%Y%m%d_%H%M%S', time.localtime()) + f'_({allAttendants}).txt'
        addText(f'{allAttendants}\n', self.chatLogsDir / self.logFileName) # chatLogに参加者名を追加
    
    def getLastContext(self):
        return self.systemContext.get()[-1]
    
    def getCalledSpeaker(self):
        lastContext = self.systemContext.get_last()
        nextSpeakerName = find_string_positions(lastContext['content'], self.attendants, lastContext['speaker'])
        if nextSpeakerName is None:
            return None
        self.speakerName = nextSpeakerName['first_word']
        nextSpeaker = self.characterAIsDict[self.speakerName]
        return nextSpeaker
        


def main():
    if not voicevoxHelthCheck(): return
    try:
        audioPlayer = WavQueuePlayer('./tmpWaveDir')
        audioPlayer.play()
        
        LLM = OpenAILLM()
        ryuseiAI = CharacterAI(LLM, "龍星", contextDB_json, 13, 1.2)
        metanAI = CharacterAI(LLM, "めたん", contextDB_json, 6, 1.2)
        tumugiAI = CharacterAI(LLM, "つむぎ", contextDB_json, 8, 1.2)
        zundamonAI = CharacterAI(LLM, "ずんだもん", contextDB_json, 7, 1.2)

        characterAIs = [ryuseiAI, metanAI, tumugiAI, zundamonAI]

        conversationTimingChecker = ConversationTimingChecker()
        chatController = ChatController(characterAIs)
        chatController.initContextAll()
        chatController.addContextAll(
            'system', "[プロデューサーとしての発言]\nあなたたちはラジオ出演者です。好きなアーティストに関してトークしてください。適宜話題は変更してください。")

        
        
        for i in range(100):
            try:
                conversationTimingChecker.check_conversation_timing_with_delay(audioPlayer.get_file_queue_length)
                chatController.getNextCharacterResponse(outputNum = 30)
                # time.sleep(10)

                if i % 8 == 0:
                    chatController.addContextAll(
                        'system', "[プロデューサーとしての発言]\n別の話題を提案して、話してください。")
                elif i % 4 == 0:
                    chatController.addContextAll(
                        'system', "[プロデューサーとしての発言]\n話の深堀をしてください。")

            except openai.error.RateLimitError:
                print("rate limit error")
                time.sleep(5)
            except openai.error.APIError:
                print("API error")
                time.sleep(5)
                
    except KeyboardInterrupt:
        audioPlayer.stop()

if __name__ == '__main__':
    main()

