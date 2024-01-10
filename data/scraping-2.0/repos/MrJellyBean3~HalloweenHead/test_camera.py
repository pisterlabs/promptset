import cv2
from pvrecorder import PvRecorder
import openai
import wave, struct 
import time
#Making a change right here
#Making 1 more change, adding to this some more


# Test chjange


# Record audio
def record_audio(return_queue):
    recorder = PvRecorder(device_index=0, frame_length=512)
    audio = []
    path = 'audio_recording.wav'
    avg_loudness=0
    t_start=time.time()
    tlt_100=time.time()
    try:
        # Start recording
        recorder.start()
        print("Starting Recording")
        while True:
            # Get audio frame
            frame = recorder.read()
            audio.extend(frame)

            # Calculate average loudness
            avg_loudness+=np.mean(np.abs(frame))*0.3
            avg_loudness*=0.7
            print("Average Loudness: ", avg_loudness,"     ", end='\r')

            # If the average loudness is below 120 for 0.35 seconds after 1 second, stop recording
            if (time.time()-t_start)>1.0:
                if avg_loudness<120:
                    if (time.time()-tlt_100)>0.35:
                        raise KeyboardInterrupt
                else:
                    tlt_100=time.time()
            else:
                tlt_100=time.time()
    except KeyboardInterrupt:
        # Stop recording and save to file
        recorder.stop()
        with wave.open(path, 'w') as f:
            f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            f.writeframes(struct.pack("h" * len(audio), *audio))
        recorder.delete()
        print("Recording Finished")
    finally:
        # Transcribe audio
        t_start=time.time()
        audio_file= open(path, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        t_end=time.time()
        print("Time to transcribe: ",t_end-t_start, " Text:",transcript["text"])

        # Return transcript to main thread to display and for functions
        return_queue.put(transcript["text"])
        return(transcript["text"])
