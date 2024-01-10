import asyncio
import tempfile
import os
import time
import chatgpt
import openai
import keyboard
import speech_recognition as sr
from pybass3 import Song
import edge_tts

communicate = edge_tts.Communicate()

async def main():
    """
    Main function
    """
    while True:
        chatgpt.login()
        if keyboard.is_pressed("q"):
            loop.stop()
        r = sr.Recognizer()

        print("Habla:")
        with sr.Microphone() as source:
            audio = r.listen(source)

        opa = True
        try:
            command = r.recognize_google(audio,language="es-ES")
            print("Comando: "+ command)

        except:
            opa = False
            print("No te entendi")
            command = ""

        if opa:
            print("Consultando con GPT-3")
            response = openai.Completion.create(
                    engine = "text-davinci-003",
                    prompt = f"{command}",
                    max_tokens=1024,
                    temperature=0.5,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            command = response["choices"][0]["text"]

            print("Procesando...")
            with tempfile.NamedTemporaryFile(suffix='.mp3') as temporary_file:
                async for i in communicate.run(command, voice='es-MX-DaliaNeural'):
                    if i[2] is not None:
                        temporary_file.write(i[2])

                song = Song(temporary_file.name)
                song.play()
                len_bytes = song.duration_bytes
                position_bytes = song.position_bytes
                print("Respondiento... presiona la tecla S para detener")
                while position_bytes < len_bytes:
                    print(song.position, song.duration)
                    if keyboard.is_pressed("s"):
                        song.stop()
                        break
                    time.sleep(1)
                    position_bytes = song.position_bytes


if __name__ == "__main__":
    openai.api_key = "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"
    asyncio.get_event_loop().run_until_complete(main())
