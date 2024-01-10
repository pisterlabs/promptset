#! /usr/bin/env python
import openai
from pydub import AudioSegment
import math
import glob
import os
import shutil

#音声ファイルの文字起こしを実行する処理
def transcriptionAudio(audioFilePath, ext, openAiKey):

    openai.api_key = openAiKey# openaiキーをセット
    makeDirPath = "transcriptData"# 分割後の音声ファイルを格納するディレクトリ名

    #既にファイルが存在する場合は削除する。
    if os.path.exists(makeDirPath):
        shutil.rmtree(makeDirPath)

    #音声分割後のファイルを保存するディレクトリを作成する。
    os.makedirs(makeDirPath)

    #whisperの容量制限を回避できるように音声ファイルを分割
    splitAudioFile(audioFilePath, ext, makeDirPath)

    path_list = glob.glob(makeDirPath + '/*.' + ext)
    name_list = []
    for i in path_list:
        file = os.path.basename(i)
        name, ext = os.path.splitext(file)
        name_list.append(name)

    fileList = sorted(name_list)
    transcript = createChatGPTSendText(fileList, makeDirPath, ext)

    #音声分割後のファイルを保存するディレクトリを削除する。
    shutil.rmtree(makeDirPath)

    return transcript

#音声ファイルを1分ごとに分割する関数
def splitAudioFile(audioFilePath, ext, makeDirPath):
    
    audio_file= AudioSegment.from_file(audioFilePath, format=ext)
    #音声ファイルの全体の長さを取得
    seconds = audio_file.duration_seconds
    print(f'音声ファイルの長さ : {seconds}秒')#コンソールでの確認用
    #オーディオファイルの分割数
    fileCount = math.ceil((seconds / 60))
    print(f'オーディオファイルの分割数 : {fileCount}')#コンソールでの確認用

    #音声ファイルを1分ごとに分割
    for i in range(fileCount):
        if i == 0:
            #音声ファイルから最初の1分を抽出する。
            audioI = audio_file[0:60000]
        else:
            #音声ファイルから最初の1分を抽出する。
            audioI = audio_file[(60000 * i):(60000 * (i + 1))]
        
        exportFileName = makeDirPath + '/' + str(i + 1) + "." + ext
        
        #抽出した部分の出力
        audioI.export(exportFileName, format=ext)

#分割後のファイルを読み込んで、chahtGPTに送る用の文章に加工する。
def createChatGPTSendText(fileList, makeDirPath, ext):
    
    text = ''#テキストの宣言
    
    for i in range(len(fileList)):
        print('whisperで文字起こししています。' + str(i + 1) + '/' + str(len(fileList)))#コンソールでの確認用
        audio_file= open(makeDirPath + '/' + str(i + 1) + ext, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        text += transcript['text']
        
    return text