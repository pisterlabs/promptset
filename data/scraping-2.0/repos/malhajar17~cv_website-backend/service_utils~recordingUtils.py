import os

import azure.cognitiveservices.speech as speechsdk
import ffmpeg
import openai
import whisper
import replicate

import constants.paths as paths
import constants.prompts as prompts
import service_utils.databaseUtils  as databaseUtils


def convert_webm_to_wav(input_path, wav_path):
    # Check if output file already exists, if so, delete it
    if os.path.exists(wav_path):
        os.remove(wav_path)
    ffmpeg.input(input_path).output(wav_path, format='wav').run()


def allowed_file(filename):
    return True



def generate_text(First_user_message,accountid,sessionID ,sequence, type,model=os.environ.get("OPEN_AI_MODULE"), temperature=0.3,isWarmingUp=False):
    print(type)
    prompt = prompts.MOHAMAD_PERSONA_PROMPT
    try:
        if type == "gpt-3.5-turbo":
            if isWarmingUp:
                messages = [{"role": "system", "content": "you are chatgpt"}, {"role": "assistant", "content": "Hi"}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1000,
                )
                return True

            prompt = prompts.MOHAMAD_PERSONA_PROMPT

            messages = [{"role": "system", "content": prompt}, {"role": "assistant", "content": First_user_message}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
            )
            databaseUtils.create_gptresponse_entry(account_id=accountid, session_id=sessionID, sequence=sequence, text=response.choices[0].message["content"])
            return response.choices[0].message["content"]

        elif type=="llama":
            if isWarmingUp:
                return True
            else:
                output = replicate.run(
                    "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
                    input={"system_prompt": prompt, "prompt": First_user_message},
                    max_new_tokens=70
                )
                llama_output = ''.join(val for val in output)
                if llama_output == '':
                    return "I'm sorry, I couldn't understand that. Could you  repeat it, please?"
                else:
                    databaseUtils.create_gptresponse_entry(account_id=accountid, session_id=sessionID, sequence=sequence, text= llama_output)
                    return llama_output


    except Exception as e:
        # Log the error for debugging
        print(f"Error encountered: {e}")
        return "I'm sorry, I couldn't understand that. Could you repeat it, please?"

def text_to_speech(text,path):
    # Creates an instance of a speech config with specified subscription key and service region.
    speech_key = os.environ.get("AZURE_COGNITIVE_TOKEN")
    service_region = "eastus"
    file_name = "gs_" + path
    path = os.path.join(os.getcwd(), paths.GENERATED_SPEECH_PATH + file_name + ".wav")

    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False, filename=path)
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Note: the voice setting will not overwrite the voice element in input SSML.
    speech_config.speech_synthesis_voice_name = "en-US-DavisNeural"
    speech_config.speech_synthesis_language = "en-US"
    # use the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesizer.speak_text(text)


def speech_to_text(data_path,accountID,sessionID,sequence):
    model = whisper.load_model(os.environ.get("WHISPER_MODEL"))
    file_name = "re_" + data_path
    path = os.path.join(os.getcwd(), paths.RECORDED_SPEECH_PATH + file_name +".wav")
    transcription = whisper.transcribe(model, path)
    databaseUtils.create_transcription_entry(accountID,sessionID,sequence,transcription["text"])
    return transcription["text"]
