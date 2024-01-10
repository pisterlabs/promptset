import re
import os
import yt_dlp
from celery import shared_task
from django.conf import settings
from .models import Subtitle
from users.models import Profile
import openai
import codecs


openai.api_key = settings.WHISPER_API_KEY


@shared_task()
def generate_subtitle(subtitle_id):
    subtitle = Subtitle.objects.get(id=subtitle_id)
    profile = Profile.objects.get(user=subtitle.user)

    audio_file_path = download_audio(subtitle)
    if audio_file_path:
        subtitle.status = 'transcribing'
        subtitle.error = ''
        subtitle.save()
        transcript = transcribe_audio(audio_file_path, subtitle)
        if transcript:
            # save this transcript into a file with title as name and .srt as extension
            cleaned_title = re.sub(r'[^a-zA-Z0-9\s]', '', subtitle.title)
            srt_file_path = f'{settings.MEDIA_ROOT}/subtitles/{cleaned_title}.srt'
            with codecs.open(srt_file_path, 'w', encoding='utf-8') as subtitle_file:
                subtitle_file.write(transcript)
            subtitle.status = 'completed'
            subtitle.error = ''
            subtitle.file_path = f'{settings.MEDIA_URL}subtitles/{cleaned_title}.srt'
            subtitle.save()
            # charge the user
            if not subtitle.is_paid:
                # if the user has enough credits
                if profile.credits >= subtitle.cost:
                    profile.credits -= subtitle.cost
                    profile.save()
                    subtitle.is_paid = True
                    subtitle.save()
                else:
                    subtitle.status = 'failed'
                    subtitle.error = 'Not enough credits'
                    subtitle.save()
        else:
            subtitle.status = 'failed'
            subtitle.save()
    else:
        subtitle.status = 'failed'
        subtitle.save()


def transcribe_audio(audio_file_path, subtitle):
    try:
        with open(audio_file_path, 'rb') as audio_file:
            transcript = openai.Audio.transcribe(
                file=audio_file,
                model="whisper-1",
                response_format="srt",
                language="en"
            )
        # delete the audio file
        os.remove(audio_file_path)

        return transcript
    except Exception as e:
        print(e)
        subtitle.status = 'failed'
        subtitle.error = 'Failed to transcribe audio'
        subtitle.save()
        return None


def download_audio(subtitle_obj):
    try:
        subtitle_obj.status = 'downloading'
        subtitle_obj.save()
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{settings.MEDIA_ROOT}/audios/%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '64',
            }],
            'quiet': True,
            'ignoreerrors': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(subtitle_obj.url, download=True)
            audio_file_path = ydl.prepare_filename(info)
            base, _ = os.path.splitext(audio_file_path)
            new_file = base + '.mp3'

        return new_file

    except Exception as e:
        print(e)
        subtitle_obj.status = 'failed'
        subtitle_obj.error = 'Failed to download audio'
        subtitle_obj.save()
        return None
