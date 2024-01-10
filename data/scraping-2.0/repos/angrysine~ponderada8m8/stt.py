from openai import OpenAI
from decouple import config
import soundfile as sf
import simpleaudio as sa
from txtai.pipeline import TextToSpeech
from txtai.pipeline import TextToSpeech
KEY = config('OPENAI_API_KEY')
client = OpenAI(api_key=KEY)



while True:
    input_text = input("Digite o nome do arquivo: ")
    if input_text == 'exit':
        break
  # Build pipeline
    tts = TextToSpeech("NeuML/ljspeech-jets-onnx")
    with open(input_text, "rb") as audio_file:
      transcript = client.audio.translations.create(
      model="whisper-1", 
      file=audio_file, 
      response_format="text"
    )

    # Generate speech
    speech = tts(transcript)

    # Write to file
    sf.write("out.wav", speech, 22050)

    # Read the file
    wave_obj = sa.WaveObject.from_wave_file("out.wav")

    # Play audio
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until audio finishes playing
    print(transcript)
    print("audio salvo em out.wav")