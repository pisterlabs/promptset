async def write(self, audio_file):
    from bot.src.utils import config
    import secrets
    import openai
    if self.api not in config.api["available_transcript"]:
        index = secrets.randbelow(len(config.api["available_transcript"]))
        self.api = config.api["available_transcript"][index]
    openai.api_key = config.api["info"][self.api]["key"]
    openai.api_base = config.api["info"][self.api]["url"]
    r = await openai.Audio.atranscribe("whisper-1", audio_file)
    return r["text"]