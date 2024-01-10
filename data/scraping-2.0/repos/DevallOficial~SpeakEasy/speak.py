import os

import azure.cognitiveservices.speech as speechsdk
import openai


def transcrever_fala():
    
    chave_assinatura = ''
    regiao = 'brazilsouth'

    
    config = speechsdk.SpeechConfig(subscription=chave_assinatura, region=regiao)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=config, language='pt-BR')

    print("Fale algo...")

    # Iniciar o reconhecimento de fala
    result = speech_recognizer.recognize_once()

    # Verificar se o reconhecimento foi bem-sucedido
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        texto = result.text
        print("Texto capturado:", texto)
        return texto

    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("Não foi possível reconhecer o áudio")
        return ""

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Reconhecimento cancelado. Motivo:", cancellation_details)
        return ""

def resumir_texto(texto):
    openai.api_key = ''
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Resuma essa reunião com os pontos principais, dê um titulo e se necessário utilize tópicos e descarte conversas paralelas: " + texto,
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        resumo = response.choices[0].text.strip()
        return "Resumo: " + resumo
    except openai.error.RateLimitError as error:
        return "Erro: " + str(error)

while True:
    
    texto_falado = transcrever_fala()
    if len(texto_falado) > 10:
        resumo = resumir_texto(texto_falado)
        print(resumo)
        break
