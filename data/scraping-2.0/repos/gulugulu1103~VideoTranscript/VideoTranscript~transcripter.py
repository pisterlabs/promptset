import os
import random
import re
import time

import dotenv
from openai import OpenAI

from audio_processor import AudioProcessor
from abc import ABC, abstractmethod


# 定义一个Transcipter的抽象类
class Transcripter(metaclass = ABC):
	@abstractmethod
	def process(self, file_name):
		pass

	@staticmethod
	@abstractmethod
	def process_all(bvid: str, segment_length: int):
		pass


class OnlineWhisper(Transcripter):
	def __init__(self, bvid: str, segment_length: int):
		"""
		初始化Transcripter类的实例。

		:param bvid: Bilibili视频的BV号。
		:param segment_length: 音频片段的长度，单位为秒。

		"""
		self.bvid = bvid
		self.segment_length = segment_length
		self.api_key = dotenv.get_key("../.env", "OPENAI_KEY")
		self.client = OpenAI(api_key = self.api_key, base_url = "https://orisound.cn/v1")

	def process(self, file_name: str):
		"""
		处理指定的文件，包括设置文件路径，处理音频，以及转录。

		:param file_name: 需要处理的文件名。
		"""
		# 设置文件路径
		file = file_name
		part_name = os.path.splitext(os.path.basename(file))[0]
		audio_file = f"../data/{self.bvid}/audios/{part_name}.mp3"
		transcript_file = f"../data/{self.bvid}/transcriptions/{part_name}.txt"
		segments_path = os.path.join(f"../data/{self.bvid}/segments/", part_name)

		# 处理音频文件
		if not os.path.exists(audio_file):
			os.makedirs(os.path.dirname(audio_file), exist_ok = True)
		AudioProcessor.extract_and_normalize_audio(file, audio_file)
		os.makedirs(segments_path, exist_ok = True)
		AudioProcessor.split_audio_by_duration(audio_file, segments_path, self.segment_length)

		# 对音频文件进行转录
		if not os.path.exists(transcript_file):
			os.makedirs(os.path.dirname(transcript_file), exist_ok = True)
		for segment_audio in os.listdir(segments_path):
			print("正在处理：" + segment_audio)
			with open(os.path.join(segments_path, segment_audio), "rb") as f:
				transcript = self.client.audio.transcriptions.create(
						model = "whisper-1",
						file = f,
						response_format = "json"
				)
			with open(transcript_file, "a", encoding = 'utf-8') as f:
				f.write(transcript.text)

	@staticmethod
	def process_all(bvid: str, segment_length: int):
		transcripter = OnlineWhisper(bvid, segment_length)
		downloads_path = f"../data/{bvid}/downloads"
		files = os.listdir(downloads_path)
		# 自定义排序函数，只考虑以"[P数字]"开头的文件名
		files.sort(
				key = lambda x: int(re.search(r'\d+', x).group()) if x.startswith('P') and re.search(r'\d+',
				                                                                                     x) else float(
						'inf'))
		for file_name in files:
			# 只对音视频类文件执行
			if file_name.endswith((".mp4", ".avi", ".flv", ".mkv", ".mov", ".wmv", ".mp3", ".wav", ".m4a")):
				file_path = os.path.join(downloads_path, file_name)
				print(file_path)
				transcripter.process(file_path)
				time.sleep(random.randint(10, 20))


import whisper


class LocalWhisper(Transcripter):
	def __init__(self, bvid: str, segment_length: int):
		"""
	    初始化LocalWhisper类的实例。

	    :param bvid: Bilibili视频的BV号。
	    :param segment_length: 音频片段的长度，单位为秒。
	    """
		self.bvid = bvid
		self.segment_length = segment_length

	def process(self, file_name: str):
		"""
        处理指定的文件，包括设置文件路径，处理音频，以及转录。
        :param file_name: 需要处理的文件名。
		"""

	@staticmethod
	def process_all(bvid: str, segment_length: int):
		"""
	    处理所有的文件。

	    :param bvid: Bilibili视频的BV号。
	    :param segment_length: 音频片段的长度，单位为秒。
	    """
# 在这里添加你的代码


if __name__ == '__main__':
	OnlineWhisper.process_all("BV1mH4y1y72b", 300)
