from datetime import datetime
from django.conf import settings
import os
# from TTS.api import TTS
from gtts import gTTS
import openai

from chat.load_speaker_module import speaker_encoder_model_twi, speaker_encoder_model_ewe


class Chatbot:
    def __init__(self, api_key):
        openai.api_key = api_key

    def get_response(self, user_input, language, input_type):
        # Define context in both Twi and English
        context_en = """
       Context: MTN Network Service Provider

        MTN is a leading network service provider that offers a wide range of services, including mobile data, voice calls, and more. Our mission is to provide our customers with seamless connectivity and exceptional customer service.

        User Queries:
        1. How can I buy MTN data bundles?
        2. Where can I locate the nearest MTN office?
        3. Can you help me with information on MTN's international roaming services?
        4. What are the steps to port my number to MTN network?
        5. Is there a self-service app available for MTN customers?
        6. What options are available for MTN postpaid users?

        Feel free to ask any questions related to MTN services, and we'll be happy to assist you!

        """

        context_twi = """
        Context: MTN Network Service Provider

        MTN yɛ ɔhaw bi mu bi a ɛwɔ nnidi mfitiase sɛ, na wɔde nkↄmmↄ no, nkↄmmↄnkↄmmↄ, nkutoo na nnidi mfitiase bi a ɛsↄ yɛ.
        Yɛn asafo no yɛ sika akokↄkↄↄ no a, yegye nnidi nnidi bi ma wↄn atoↄ yↄnka no, na yↄn atete yie no adwuma.

        Anidaso a Ɔbarima/Ɔbea no awↄ.
        1. Me dↄ MTN dↄta baablu bↄmmobi?
        2. Ɛhwↄ me soro MTN ofis firi ha?
        3. Wo nni sika akawↄ, anaa wopɛ sika akawↄ wↄ MTN afa no?
        4. Adwuma a wↄde nnidi no ↄhia no asomdwoe mu sika akawↄ? 
        5. Saa ara no, MTN mma adwuma no ↄde wo app akawↄ.
        6. Afei dↄn sika akawↄ bɛn wo dↄm nni ho?

        Saa nti, monka adↄyↄ ase a, ɛno yie no.
       
        """

        context_ewe = """

            Context: MTN Network Service Provider

            MTN wɔ kwan no a ɛma nnidi mfitiase sɛ, nso ɛka ntwam. Sɛ ɛbɛma wo, moblɛ dɛta, asɔ wɔ nni kasa, na nsɛm bi a ɛbɛma wo. Yɛ kyerɛwfo a ɛma wɔbɛtumi aka ɔkwan, na yɛn atete nso ahyɛdeɛ.

            Ɔbarima/Ɔbea Anidaso:
            1. Menka sika nsu ma MTN nombɛ no, sɛnea ɛbɛyɛ a?
            2. Ƌfeme de la miwo MTN ofis nyanya kple?
            3. Ɛyɛ dometɔwo ɖeka nuto la gɔme MTN's international roaming afe?
            4. Ɛyi wo de nutodzi wo MTN network nutolu ɖekawo gake?
            5. Ɛsiwɔsia yeye dzi wo ame do ŋu nɔ na ɖekawo fawo MTN nombɛ nɔ wɔe?
            6. Ata wotutuwo kple MTN postpaid nombɛ nɔ nuto wo me?

            Wobɛkɔ ɛpam ɔbarima/Ɔbea Anidaso mu ase, na yɛbɛsie wo.        

        """

        twi_dict = {
            "hello": "Hello, yɛma wo akwaaba ba MTN customer service, yɛbɛyɛ dɛn aboa wo nnɛ?",
            "mɛyɛ dɛn atɔ mtn data":"Yɛda mo ase sɛ moakɔ yɛn nkyɛn. Sɛ wopɛ mmoa wɔ airtime a wobɛtɔ anaasɛ credit ma wo MTN nɔma a, yɛsrɛ wo frɛ '*143#' fi wo fon so. Sɛ worehyia nsɛmnsɛm wɔ wo nkrataahyɛ ho a, wobɛtumi akɔ yɛn adetɔfoɔ mmoa nkyɛn",
            "ɛhe na metumi ahwehwɛ mtn office":"",
            "Me ntumi nnya intanɛt nkitahodi mfiri me sim card so":"Yɛpa kyɛw wɔ ɔhaw no ho. Yɛsrɛ wo hwɛ hu sɛ wo data nhyehyɛe no yɛ adwuma na wɔahyehyɛ wo mfiri no yiye sɛnea ɛbɛyɛ a ɛbɛkɔ intanɛt so. Sɛ asɛm no kɔ so a, yɛsrɛ wo, di yɛn atɔfo adwumayɛfo kuw no nkitaho na woanya mmoa foforo. Meda wo ase.",
                    
                    }
        
        ewe_dict = {
            "hello": "Mido gbe na wò, míexɔ wò nyuie ɖe MTN ƒe asisiwo ƒe dɔwɔƒe, aleke míate ŋu akpe ɖe ŋuwò egbea?",
            "aleke mawɔ aƒle mtn data bundle":"Akpe na mi be miedo asi ɖe mía gbɔ. Ne èdi kpekpeɖeŋu le yameʋuɖoɖo ƒeƒle alo credit na wò MTN xexlẽdzesi la, taflatse ƒo '*143#' tso wò fon dzi. Ne kuxiwo le fu ɖem na wò le wò nudɔdɔ ŋu la, àte ŋu aɖo míaƒe asisiwo ƒe kpekpeɖeŋunadɔa gbɔ",
            "afi kae mate ŋu akpɔ mtn ɔfis le":"",
            "Nyemete ŋu xɔ internet kadodo tso nye sim card dzi o":"Míeɖe kuku ɖe fuɖenamea ta. Taflatse kpɔ egbɔ be wò nyatakakawo ƒe ɖoɖoa le dɔ wɔm eye be woɖo wò mɔ̃a nyuie be wòate ŋu age ɖe internet dzi. Ne nyaa gakpɔtɔ li la, taflatse te ɖe míaƒe asisiwo ƒe dɔwɔƒea ŋu hena kpekpeɖeŋu bubuwo. Akpe na wò.",
                    
             }

        # Select the appropriate context based on the chosen language
        if language == "English":
            context = context_en
        elif language == "Twi":
            context = context_twi
        elif language == "Ewe":
            context = context_ewe
        else:
            context = context_en  # Default to English if the language is not recognized

        # Create a prompt that includes the user's input, context, and desired language
        prompt = f"{context}\nLanguage: {language}\nUser Input: {user_input}\nResponse:"

        response = ""

        if language == "English":
        # Make a request to the OpenAI GPT API to generate a response
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",  # You can choose the engine that suits your needs
                prompt=prompt,
                max_tokens=50,  # Adjust the max_tokens as needed
                n=1  # You can generate multiple responses and choose the best one if needed
            )

        elif language == "Twi":
            response = twi_dict.get(user_input)

        elif language == "Ewe":
            response = ewe_dict.get(user_input)

        chatbot_reply = ""

        if language == "English": 
            chatbot_reply = response.choices[0].text.strip()
        else:
            chatbot_reply = response

        audio_response_path = ""
        if input_type == "voice":
            audio_response_path = text_to_audio(chatbot_reply, language)
        elif input_type == "text":
            audio_response_path = chatbot_reply

        return audio_response_path


def text_to_audio(text, language):
    # Convert the chatbot response text to an audio file
    final_audio_response_path = ""
    if language == "English":
        final_audio_response_path = text_to_audio_en(text)

    elif language == "Twi":
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_response_filename = f"audio_response_{timestamp}.mp3"

    # Define the full path to save the audio file in the media directory
        audio_response_path = os.path.join(
            settings.MEDIA_ROOT, audio_response_filename)

        # api = TTS("tts_models/tw_asante/openbible/vits")
        speaker_encoder_twi = speaker_encoder_model_twi
        # speaker_encoder_ewe = speaker_encoder_model_ewe

        # speaker_encoder_twi.tts_with_vc_to_file(
        #     text,
        #     speaker_wav="/Users/m1macbookpro2020/Desktop/samuel/final year project/VCABuddy/chat/speaker.mp3",
        #     file_path=audio_response_path
        # )
        speaker_encoder_twi.tts_to_file(
            text,
            # speaker_wav="/Users/m1macbookpro2020/Desktop/samuel/final year project/VCABuddy/chat/speaker.mp3",
            file_path=audio_response_path
        )
        final_audio_response_path = os.path.relpath(
            audio_response_path, settings.MEDIA_ROOT)

    elif language == "Ewe":
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_response_filename = f"audio_response_{timestamp}.mp3"

    # Define the full path to save the audio file in the media directory
        audio_response_path = os.path.join(
            settings.MEDIA_ROOT, audio_response_filename)

        # api = TTS("tts_models/tw_asante/openbible/vits")
        speaker_encoder_ewe = speaker_encoder_model_ewe
        # speaker_encoder_ewe = speaker_encoder_model_ewe

        speaker_encoder_ewe.tts_to_file(
            text,
            # speaker_wav="/Users/m1macbookpro2020/Desktop/samuel/final year project/VCABuddy/chat/speaker.mp3",
            file_path=audio_response_path
        )
        # speaker_encoder_ewe.tts_with_vc_to_file(
        #     text,
        #     speaker_wav="/Users/m1macbookpro2020/Desktop/samuel/final year project/VCABuddy/chat/speaker.mp3",
        #     file_path=audio_response_path
        # )
        final_audio_response_path = os.path.relpath(
            audio_response_path, settings.MEDIA_ROOT)

    # Return the relative path to the audio file
    return final_audio_response_path


def text_to_audio_en(text):
    # Convert the chatbot response text to an audio file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    audio_response_filename = f"audio_response_{timestamp}.mp3"

    # Define the full path to save the audio file in the media directory
    audio_response_path = os.path.join(
        settings.MEDIA_ROOT, audio_response_filename)

    audio_response = gTTS(text)
    audio_response.save(audio_response_path)

    # Return the relative path to the audio file
    return os.path.relpath(audio_response_path, settings.MEDIA_ROOT)


if __name__ == "__main__":
    # Replace with the user's text input
    user_input = "Hello, chatbot. How can I assist you?"
    chatbot = Chatbot()
    chatbot_response = chatbot.get_response(user_input)
    audio_response_path = text_to_audio(chatbot_response)

    # Now you can send the 'audio_response_path' to the frontend for playback.
