from openai import OpenAI

# 下面这句话将在执行from tts import tts时执行一次
# 在随后调用tts(input, outputPath)时，不会重复执行
client = OpenAI()

def tts(input, outputPath):
    print("tts: " + input)
    response = client.audio.speech.create(
        # model="tts-1",
        model="tts-1-hd", # 这个模型的效果更好，但是速度更慢
        voice="nova",
        input=input,
        speed=0.8
    )

    response.stream_to_file(outputPath)