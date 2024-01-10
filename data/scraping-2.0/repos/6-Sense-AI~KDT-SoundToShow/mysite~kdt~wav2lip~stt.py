from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import pysrt
import datetime
import openai

def add_subtitles(video_path, srt_path, output_path):
    video = VideoFileClip(video_path)
    
    subs = pysrt.open(srt_path, encoding='utf-8')
    subtitles = []
    for sub in subs:
        start_time = sub.start.to_time()
        end_time = sub.end.to_time()
        text = sub.text
        subtitle_clip = TextClip(text, font='Arial-Bold', fontsize=24, color='white')
        subtitle_clip = subtitle_clip.set_position(('center', 'bottom')).set_duration(
            (datetime.timedelta(hours=end_time.hour, minutes=end_time.minute, seconds=end_time.second) -
             datetime.timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second))
            .total_seconds()
        )
        subtitle_clip = subtitle_clip.set_start(
            (datetime.timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second))
            .total_seconds()
        )
        subtitles.append(subtitle_clip)
    
    final_clip = CompositeVideoClip([video.set_audio(None)] + subtitles)

    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

video_path = "video.mp4"
srt_path = "subtitles.srt"
output_path = "output.mp4"

add_subtitles(video_path, srt_path, output_path)