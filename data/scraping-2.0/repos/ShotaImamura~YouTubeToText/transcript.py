
import whisper
import os
import openai

# 必要に応じて変更
name = "rkmtlabMeeting616"

# Define the API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# audio fileの分割済みのファイルパスのリストを取得
audio_short_file_list = []
for file in os.listdir("audio/{}".format(name)):  
    audio_short_file_list.append("audio/{}/{}".format(name, file))

print(audio_short_file_list)

#audio/nameディレクトリがなければ作成
if not os.path.exists("transcript/{}".format(name)):
    os.mkdir("transcript/{}".format(name))


for audio_file_path in audio_short_file_list:
    audio_file_obj = open(audio_file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file_obj)
    print(transcript["text"])

    result = transcript["text"]

    # Do something with the transcription result...
    # 結果をテキストファイルに書き込む
    num = audio_file_path.split("/")[-1].split(".")[0]
    with open("transcript/{}/{}.txt".format(name,num), mode="w") as f:
        f.write(result)