import openai, whisper, json, aiofiles, os, asyncio

def sync_api_call(audio_path):
    try:
        with open(audio_path, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1",f )
            return transcript.get("text", "")
    except Exception as e:
        print(f"Error processing API: {e}")

async def transcribe(audio_path, transcript_path, api=True, model=None):
    try:
        if api:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, sync_api_call, audio_path)
        else:
            text = model.transcribe(audio_path)
        
        async with aiofiles.open(f"{transcript_path}.txt", mode="w") as file:
            await file.write(text)
    except Exception as e:
        print(f"Error processing files: {e}")

async def transcribe_directory(audio_folder, transcript_path, api=True, model=None):
    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            audio_path = os.path.join(audio_folder, filename)
            transcript_file = os.path.splitext(filename)[0]
            await transcribe(audio_path, os.path.join(transcript_path, transcript_file), api=api, model=model)