import os
import openai
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

SPEECH_KEY="__your__azure__speech__key__"
SPEECH_REGION="francecentral"
SPEECH_LANGUAGE="fr-FR"
OPENAI_KEY="__your__openai__key__"

CS_KEY = "__your__cognitive__service__key__"
CS_ENDPOINT = "https://___your__endpoint__.cognitiveservices.azure.com/"

def text2speech(texte):
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language=SPEECH_LANGUAGE

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    result = speech_synthesizer.speak_text_async(texte).get()

    with open("file.wav", "wb") as audio_file:
        audio_file.write(result.audio_data)

def sentimental_analysis(text):
    credential = AzureKeyCredential(CS_KEY)
    text_analytics_client = TextAnalyticsClient(CS_ENDPOINT, credential)
    result = text_analytics_client.analyze_sentiment([text])[0]

    if (result.sentiment =="negative"):
        print("   => Le client n'est pas content :(")
    else:
        print("   => Le client est content :)")

def speech2text():
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language=SPEECH_LANGUAGE

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("  Q: {}".format(speech_recognition_result.text))

        text_micro = speech_recognition_result.text
        return text_micro

    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        return ""
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
        return ""
    else:
        return ""

def qa(prompt_initial):
    openai.api_key = OPENAI_KEY

    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_initial,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )

    return response

### CODE ###
context = "Tu es un commercial dans la vente de marteaux pour la société HAMMERIZED. \
    Tu as lu toutes les astuces du livre de Vendeur d'elites de Michel AIGULAR pour avoir toutes les astuces sur la vente, \
        tu te serts de ses astuces pour convaincre les futurs clients.\nTu es actuellement en communication avec un client nommé John DOE \
            et tu as établi que c'était un homme de 50ans qui a prévu de refaire sa maison. Tu as aussi identifié qu'il avait un profil DISC Stable. \
                Fais tout ce qu'il faut pour qu'il achete un marteau. Tu ne dois pas évoquer le fait que tu as analysé son profil.\n\n\
                    Q: Qui est le leader de la vente de marteau ?\nA: La société HAMMERIZED est le leader mondial de vente de marteau\n\n\
                        Q: Depuis combien de temps la société existe-t-elle ?\nA: La société a 20 ans\n\n\
                            Q: Qui construit les meilleurs marteaux ?\n\A: Les meilleurs marteaux sont construis par la société HAMMERIZED, ils permettent de ne pas avoir d'ampoule aux mains meme après un usage intensif toute la journée à l'aide du revetement innovant sur le manche.\n\nQ: Quels types de marteaux sont vendus par HAMMERIZED ?\nA: Cinq types de marteau sont disponibles, Le marteau classique ou marteau de menuisier, Le marteau rivoir, Le marteau d'électricien Le marteau arrache-clous ou marteau américain,  et le marteau de vitrier.\n\nQ: Qu'y a-t-il en stock ?\nA: Il y a 10 unités pour chaque modèle et il reste 1 seul marteau de vitrier.\n\nQ: Pouvez vous m'indiquer le marteau le plus adapté pour rénover ma maison ?\nA: Pour rénover votre maison, je vous recommande le marteau classique ou marteau de menuisier. Il est très polyvalent et peut être utilisé pour enfoncer des clous, démonter des meubles et autres travaux de menuiserie. Il est également très robuste et durable, ce qui en fait un excellent choix pour les trav\n\n\
                                Q: Quel est le prix d'un marteau ?\nA: Le prix d'un marteau est de 19€.\n\n\
                                    Q: En avez vous en stock ?\nA: Oui, nous avons 10 unités en stock. Je peux vous offrir un prix spécial si vous achetez plusieurs marteaux."


print("Parle dans ton micro: ")

while(1):
    text_micro = speech2text()
    if (text_micro != ""):
        sentimental_analysis(text_micro)
    
        context = context + "\n\nQ: " + text_micro + "\nA: "
        response = qa(context)
        context = context + response["choices"][0]["text"]
        print("  A:" +  response["choices"][0]["text"])

        text2speech(response["choices"][0]["text"])
        print("")