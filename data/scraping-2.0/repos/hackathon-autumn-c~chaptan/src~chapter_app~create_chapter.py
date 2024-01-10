#celery関係
from __future__ import absolute_import, unicode_literals

# 共通　ファイルパス設定用
from django.conf import settings
from pathlib import Path
# 動画圧縮用
import subprocess
# 文字起こし用
from faster_whisper import WhisperModel
# チャプター生成用
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

#celery関係
from celery import shared_task
# User,Chapterのデータベースを操作
from .models import User,Chapter

# メール送信モジュール
from django.core.mail import send_mail

#S3へアップロードboto3
import boto3

media_root = str(settings.MEDIA_ROOT)
# media_url = str(settings.MEDIA_URL)
bucket_name = str(settings.AWS_STORAGE_BUCKET_NAME)
media_url = "https://" + bucket_name + ".s3.ap-northeast-1.amazonaws.com/storage"

"""保存先のディレクトリ構成
src/storage/viedos→アップロードされた動画を保存
src/storage/comp_videos→圧縮された動画を保存
src/storage/transcriptions→文字起こしテキストファイルを保存
"""

# 動画保存関数
# 動画ファイルと動画タイトルを渡すと、動画ファイルを保存して保存先のパスを返す
def save_video(video_file, video_title):
    video_path = f"{media_root}/videos/{video_title}.mp4"
    # ファイルシステムに保存
    with open(video_path, 'wb+') as destination:
        for chunk in video_file.chunks():
            destination.write(chunk)

    # ローカルの動画ファイルをS3に保存
    upload_to_s3(video_path, f"storage/videos/{video_title}.mp4")
    video_path = f"{media_url}/videos/{video_title}.mp4"

    return video_path


# 動画圧縮関数
# 動画ファイルのパスと動画タイトルを与えると、動画を圧縮して圧縮動画の保存先パスを返す
# celeryで処理する関数に設定
@shared_task
def comp_mp4(user_id, video_path, video_title):
    try:
        # Chapterデータベースから動画タイトルをもとにデータを取得し、chapter_dataとstatusを上書き保存
        user = User.objects.get(pk=user_id)
        user_email = user.email
        chapter = Chapter.objects.get(video_title=video_title)
        chapter.status = '文字起こし中'
        chapter.save()

        # faster-whisperで文字起こし
        transcription_path = faster_whisper(video_path, video_title)
        print('文字起こし完了')

        # ローカルのtranscriptionファイルをS3に保存
        upload_to_s3(transcription_path, f"storage/transcriptions/trans_{video_title}.txt")
        transcription_url = f"{media_url}/transcriptions/trans_{video_title}.txt"

        # 音声テキストファイルからテキストデータを読み込み
        with open(transcription_path, encoding="utf-8_sig") as f:
            state_of_the_union = f.read()
        # Chapterデータベースから動画タイトルをもとにデータを取得し、chapter_dataとstatusを上書き保存
        user = User.objects.get(pk=user_id)
        user_email = user.email
        chapter = Chapter.objects.get(video_title=video_title)
        chapter.status = 'チャプター生成中'
        chapter.chapter_data = state_of_the_union
        chapter.save()

        # メール送信
        # """題名"""
        subject = 'チャプたん通知（文字起こし）'
        # """本文"""
        message_title = 'チャプたんで動画「' + video_title + '」の文字起こしが完了しました。'
        message = message_title
        # """送信元メールアドレス"""
        from_email = "hackathon0701@gmail.com"
        # """宛先メールアドレス"""
        recipient_list = [
            user_email
        ]
        send_mail(subject, message, from_email, recipient_list, fail_silently=False)

        # チャプター生成関数
        chapter_text = create_chap(transcription_path, video_title)
        print('チャプター生成完了')

        # Chapterデータベースから動画タイトルをもとにデータを取得し、chapter_dataとstatusを上書き保存
        chapter = Chapter.objects.get(video_title=video_title)
        chapter.chapter_data = chapter_text
        chapter.status = '完了'
        chapter.save()

        # メール送信
        # """題名"""
        subject = 'チャプたん通知（チャプター生成完了）'
        # """本文"""
        message_title = 'チャプたんで動画「' + video_title + '」のチャプター生成が完了しました。'
        message = message_title
        # """送信元メールアドレス"""
        from_email = "hackathon0701@gmail.com"
        # """宛先メールアドレス"""
        recipient_list = [
            user_email
        ]
        send_mail(subject, message, from_email, recipient_list, fail_silently=False)

        # #動画の圧縮・保存
        # comp_video_path = f"{media_root}/comp_videos/comp_{video_title}.mp4"
        # # 動画の高さを720で固定
        # subprocess.call(f'ffmpeg -i "{video_path}" -crf 36 -vf scale=-2:360 "{comp_video_path}"', shell=True)
        # print('動画圧縮完了')
        # # Chapterデータベースから動画タイトルをもとにデータを取得し、chapter_dataとstatusを上書き保存
        # user = User.objects.get(pk=user_id)
        # user_email = user.email
        # chapter = Chapter.objects.get(video_title=video_title)
        # chapter.status = '動画圧縮・保存も完了'
        # chapter.video_file_path = f"/storage/comp_videos/comp_{video_title}.mp4"
        # chapter.save()

        # # メール送信
        # # """題名"""
        # subject = 'チャプたん通知（動画圧縮ALL完了）'
        # # """本文"""
        # message_title = 'チャプたんで動画「' + video_title + '」の圧縮が完了しました。'
        # message = message_title
        # # """送信元メールアドレス"""
        # from_email = "hackathon0701@gmail.com"
        # # """宛先メールアドレス"""
        # recipient_list = [
        #     user_email
        # ]
        # send_mail(subject, message, from_email, recipient_list, fail_silently=False)

    except Exception as e:
        # Chapterでエラーが起きた際の例外処理
        user = User.objects.get(pk=user_id)
        user_email = user.email
        chapter = Chapter.objects.get(video_title=video_title)
        chapter.status = 'Celeryエラー'
        # chapter.video_file_path = f"/storage/comp_videos/comp_{video_title}.mp4"
        chapter.save()

        # メール送信
        # """題名"""
        subject = 'チャプたん通知（エラー）'
        # """本文"""
        message_title = 'チャプたんで動画「' + video_title + '」の処理中にエラーが発生しました。'
        message = message_title
        # """送信元メールアドレス"""
        from_email = "hackathon0701@gmail.com"
        # """宛先メールアドレス"""
        recipient_list = [
            user_email
        ]
        send_mail(subject, message, from_email, recipient_list, fail_silently=False)
        return e

#秒数を時間、分、秒に変換する関数を作成 
def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
  
# 文字起こし関数
# 圧縮動画のファイルパスと動画タイトルを与えると、文字起こしをしてテキストのファイルパスを返す
def faster_whisper(comp_video_path, video_title):
    transcription_path = f"{media_root}/transcriptions/trans_{video_title}.txt"

    model_size = "medium"
    model = WhisperModel(model_size, device="auto", compute_type="float32")

    ####タイムスタンプ付き、テキストのみ書き出し####
    segments, info = model.transcribe(comp_video_path, beam_size=5, temperature=1.0, language="ja")

    with open(transcription_path, 'w',encoding="utf-8") as f:
        for segment in segments:
            time_formatted = seconds_to_hms(segment.start)
            print(time_formatted)
            f.write(f"[{time_formatted}] {segment.text}\n")

    return transcription_path


# チャプター生成関数
# 文字起こしテキストのファイルパスと動画タイトルを与えると、チャプターテキストを返す
def create_chap(transcription_path, video_title):
    # 音声テキストファイルからテキストデータを読み込み
    with open(transcription_path, encoding="utf-8_sig") as f:
        state_of_the_union = f.read()

    # chunk_sizeなど
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 4000,
        chunk_overlap  = 100,
        length_function = len,
        is_separator_regex = False,
    )

    texts = text_splitter.create_documents([state_of_the_union])

    # 言語モデルとしてOpenAIのモデルを指定
    llm = OpenAI(model_name="gpt-4")

    # プロンプト文
    template = """
    次の文章は時間とセリフが書いてあるシナリオです。
    この内容に最も適した見出しを作り、一番最初の時間と見出しを回答して下さい。
    回答は [時間] 見出し という形式でお願いします。「{original_sentences}」
    """

    # プロンプトのテンプレート内にあるチャプター分け前のテキストを変数として設定
    prompt = PromptTemplate(
        input_variables=["original_sentences"],
        template=template,
    )

    # プロンプトを実行させるチェーンを設定
    llm_chain = LLMChain(llm=llm, prompt=prompt,verbose=True)

    chapter_text = ''
    # for文で分割した各テキストに対しチェーンを実行
    # 実行結果をoutput.txtに出力
    for original_sentences in texts:
        response = llm_chain.run(original_sentences)
        chapter_text += f"{response}\n"

    return chapter_text


def upload_to_s3(local_path, s3_path):
    """
    ローカルファイルをS3にアップロードする関数
    :param local_path: ローカルのファイルパス
    :param s3_path: S3に保存するパス
    """
    # S3リソースオブジェクトを作成
    s3 = boto3.resource(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
    )

    # ファイルをアップロード
    bucket = s3.Bucket(settings.AWS_STORAGE_BUCKET_NAME)
    bucket.upload_file(local_path, s3_path)
