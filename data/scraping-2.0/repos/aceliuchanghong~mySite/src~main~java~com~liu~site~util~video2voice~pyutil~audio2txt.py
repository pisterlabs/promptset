import os
import sys
import openai
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def getSrtFile(audio_file, prompt, srcpath="../mp3/",
               despath="../srt/"):
    proxyHost = "127.0.0.1"
    proxyPort = 10809
    proxies = {
        "http": f"http://{proxyHost}:{proxyPort}",
        "https": f"http://{proxyHost}:{proxyPort}"
    }
    openai.proxy = proxies
    openai.api_key = os.getenv("OPENAI_API_KEY")

    realFilePath = srcpath + audio_file
    try:
        if not os.path.exists(realFilePath):
            print("Err not exists:" + realFilePath)
            return
        file = open(realFilePath, "rb")
        transcript = openai.Audio.transcribe("whisper-1", file, response_format="srt",
                                             prompt=prompt)
        with open(despath + audio_file + ".srt", 'w') as f:
            print("Srt fileName is:" + audio_file + ".srt")
            f.write(transcript)
    except Exception as e:
        print("Input Error:", e)


def getSrtFileWithoutProxy(audio_file, prompt, srcpath="../mp3/",
                           despath="../srt/"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    realFilePath = srcpath + audio_file
    try:
        if not os.path.exists(realFilePath):
            print("Err not exists:" + realFilePath)
            return
        file = open(realFilePath, "rb")
        transcript = openai.Audio.transcribe("whisper-1", file, response_format="srt",
                                             prompt=prompt)
        with open(despath + audio_file + ".srt", 'w') as f:
            print("Srt fileName is:" + audio_file + ".srt")
            f.write(transcript)
    except Exception as e:
        print("Input Error:", e)


# getSrtFile("saveHeart.mp3", "this is an audio about hearts which calling on people to protect their hearts")
getSrtFile("WeChat_20231007161725.mp3", "教学-复盘")

# fileName = sys.argv[1]
# prompt = sys.argv[2]
# srcpath = sys.argv[3]
# despath = sys.argv[4]
# if 3 > len(sys.argv) or 5 < len(sys.argv):
#     print("请至少输入2个参数,最多4个:1.文件名字,2.prompt,3.srcpath,4.despath")
# elif 3 == len(sys.argv):
#     getSrtFile(fileName, prompt)
# elif 4 == len(sys.argv):
#     getSrtFile(fileName, prompt, srcpath)
# elif 5 == len(sys.argv):
#     getSrtFile(fileName, prompt, srcpath, despath)
