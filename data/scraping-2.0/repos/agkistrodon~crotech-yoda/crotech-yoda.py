import os
import azure.cognitiveservices.speech as speechsdk
import re
import openai
from openai import AzureOpenAI

client = AzureOpenAI()

speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")

audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

# Locale for speaker's language.
speech_config.speech_recognition_language="en-US"
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# Define voice that responds on behalf of Azure OpenAI.
speech_config.speech_synthesis_voice_name='en-US-JasonNeural'

# Create speech synthesizer
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

# Text to speech sentence end marks
tts_sentence_end = [ ".", "!", "?", ";", "。", "！", "？", "；", "\n" ]

# Variables
assistant_context = "You are Yoda. Talk exactly like him, in both style and content, and limit your response length to under 35 words."
conversation_content = [{"role": "system", "content": assistant_context}]

def convert_to_hmm(text):
    # Use case-insensitive regular expression to match variations of hrrrm, hrrm, hrrrmmm, etc.
    pattern = re.compile(r'hr*m+', re.IGNORECASE)

    # Replace matched patterns with 'Hmm'
    return pattern.sub('Hmm', text)

# Prompts Azure OpenAI with a request and synthesizes the response.
def ask_openai(prompt):
    # Ask Azure OpenAI in streaming way
    conversation_content.append({"role": "user", "content": prompt})
    response =  client.chat.completions.create(model="gpt-4", messages=conversation_content, max_tokens=200, stream=True)
    collected_messages = []
    last_tts_request = None

    # iterate through the stream response stream
    response_text = ""
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is not None:
            response_text += chunk_message
            collected_messages.append(chunk_message)  # save the message
            if chunk_message in tts_sentence_end: # sentence end found
                text = ''.join(collected_messages).strip() # join the recieved message together to build a sentence
                if text != '': # if sentence only have \n or space, we could skip
                    text = convert_to_hmm(text)
                    print(f"Speech synthesized to speaker for: {text}")

                    ssml_text = """
		    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
			    <voice name="en-US-TonyNeural">
                               <prosody rate="+10%">
			        <mstts:express-as style="whispering" styledegree="1.5">
		                """ + text + """
  		                </mstts:express-as>
                               </prosody>
			    </voice>
		    </speak>
                    """
                    last_tts_request = speech_synthesizer.speak_ssml_async(ssml_text)
                    collected_messages.clear()
    conversation_content.append({"role": "assistant", "content": response_text})
    if last_tts_request:
        last_tts_request.get()

# Continuously listens for speech input to recognize and send as text to Azure OpenAI
def chat_with_open_ai():
    global conversation_content
    ask_openai("Introduce yourself in under 10 words.")
    conversation_content = [ conversation_content[0] ]
    while True:
        print("Listening. Say 'Stop' or press Ctrl-Z to end the conversation.")
        try:
            # Get audio from the microphone and then send it to the TTS service.
            speech_recognition_result = speech_recognizer.recognize_once_async().get()

            # If speech is recognized, send it to Azure OpenAI and listen for the response.
            if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                if speech_recognition_result.text == "Stop.":
                    speech_synthesizer.speak_text_async("Conversation ended.").get()
                    print("Conversation ended.")
                    break
                print("Recognized speech: {}".format(speech_recognition_result.text))
                ask_openai(speech_recognition_result.text)
            elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
                print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
            elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speech_recognition_result.cancellation_details
                print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print("Error details: {}".format(cancellation_details.error_details))
        except EOFError:
            break

# Main

try:
    chat_with_open_ai()
except Exception as err:
    print("Encountered exception. {}".format(err))
