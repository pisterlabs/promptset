from env import Env
from customtime.customtime_service import CustomTime

import openai
import os
from gtts import gTTS


class VoicePrompt:

    openai.organization = Env.OPENAI_ORG_ID
    openai.api_key = Env.OPENAI_API_KEY

    @classmethod
    def get_prompt_from_audio(cls, audio_path: str) -> str:
        audio_file = open(audio_path, 'rb')
        prompt = openai.Audio.transcribe('whisper-1', audio_file)['text']

        return prompt

    @classmethod
    async def text_to_speech(cls, text: str):
        audio_reply_path = Env.VOICE_DIR + 'reply.wav'
        tts = gTTS(text=text, lang=Env.VOICE_LANGUAGE, slow=False)
        tts.save(audio_reply_path)

        time_code = CustomTime.get_time_code()
        audio_new_reply_path = f'{Env.VOICE_DIR}voice_note_{time_code}.wav'

        if Env.VOICE_MODEL_PATH == '':
            os.rename(audio_reply_path, audio_new_reply_path)
        else:
            cmd = f'svc infer {audio_reply_path} ' \
                  f'-c {Env.VOICE_CONFIG_PATH} -m {Env.VOICE_MODEL_PATH} -na -t {Env.VOICE_TRANSPOSE}'
            os.system(cmd)
            os.rename(Env.VOICE_DIR + 'reply.out.wav', audio_new_reply_path)

        return audio_new_reply_path
