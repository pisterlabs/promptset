import pyaudio
import wave
import threading
import openai
import os
os.environ["OPENAI_API_KEY"] = "sk-"
openai.api_key = os.getenv("OPENAI_API_KEY")

class AudioRecorder:
    def __init__(self, output_file, sample_rate=44100, chunk_size=1024, channels=1):
        self.output_file = output_file
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_format = pyaudio.paInt16
        self.frames = []
        self.is_recording = False
        self.audio = pyaudio.PyAudio()

    def start_recording(self):
        self.is_recording = True
        self.stream = self.audio.open(format=self.audio_format, channels=self.channels,
                                      rate=self.sample_rate, input=True,
                                      frames_per_buffer=self.chunk_size,
                                      stream_callback=self.callback)

        print("开始录音...")

        self.stream.start_stream()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            wave_file = wave.open(self.output_file, 'wb')
            wave_file.setnchannels(self.channels)
            wave_file.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wave_file.setframerate(self.sample_rate)
            wave_file.writeframes(b''.join(self.frames))
            wave_file.close()

            print("录音结束.")

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.frames.append(in_data)
            return in_data, pyaudio.paContinue
        else:
            return in_data, pyaudio.paComplete

def manual_recording():
    # 创建录音对象
    recorder = AudioRecorder("recording.wav")

    # 启动录音线程
    recording_thread = threading.Thread(target=recorder.start_recording)
    recording_thread.start()

    # 等待用户按下回车键停止录音
    input("按下回车键停止录音...\n")

    # 停止录音
    recorder.stop_recording()

# 调用手动录音函数
manual_recording()
audio_file= open("recording.wav", "rb")
response = openai.Audio.transcribe("whisper-1", audio_file)
# 提取出转录文本
print(response)
transcript = response.text
# 将转录结果写入到文本文件
with open('transcription.txt', 'w') as f:
    f.write(transcript)

total_tokens = 0
message = {"role": "user", "content":transcript}
messagess = [message]
response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messagess)
total_tokens += response['usage']['total_tokens']
finish_reason = response['choices'][0]['finish_reason']

if finish_reason == "stop":
    print(f"正常返回，目前总共花费{total_tokens}字节")
elif finish_reason == "length":
        print("字符最高上限，重开吧")
elif finish_reason == "content_filter":
        print("输入了逆天玩意儿被屏蔽了（流汗黄豆）")
elif finish_reason == "null":
        print("未知错误")
assistant_output = response['choices'][0]['message']['content']
print(assistant_output)
with open('airespose.txt', 'w') as f:
    f.write(assistant_output)