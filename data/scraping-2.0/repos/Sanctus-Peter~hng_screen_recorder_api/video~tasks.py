import time
import openai
from .models import RecordingSession

from .utils import combine_chunks


def transcribe_video(recording_id):
    try:
        recording_session = RecordingSession.objects.get(recording_id=recording_id)
    except RecordingSession.DoesNotExist:
        return

    # Simulate transcription process
    video_file = combine_chunks(recording_id)
    transcript = openai.Audio.transcribe('whisper-1', open(video_file, 'rb'))
    time.sleep(10)  # Simulate a 10-second transcription

    recording_session.metadata = transcript.get('text', '')
    recording_session.status = 'completed'
    recording_session.save()
