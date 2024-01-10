import openai
import glob

# Loop through all files with pattern chunk*.mp3
for audio_filename in sorted(glob.glob("chunk*.mp3")):
    try:
        print(audio_filename)
        with open(audio_filename, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                file=audio_file,
                model="whisper-1",
                prompt="The speaker uses filler words a lot. Include those filler words such as 'uhh', etc.",
                language="en"
            )
            text = transcript["text"]
            # Construct the txt filename based on the mp3 filename
            txt_filename = audio_filename.replace(".mp3", ".txt")
            with open(txt_filename, 'w') as file:
                file.write(text)
            print(text)
    except Exception as e:
        print(f"An error occurred with {audio_filename}: {e}")

