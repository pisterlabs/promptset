#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa
#
# Copyright 2023 MangDang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description:
# This script transcribes Audio input using OpenAI STT and publish the transcribed text back to ROS2.
# The input volume can be re-scaled for robots with low output volume such as Mini Pupper v2.
# It also supports continuous audio input and real-time transcription by checking the GPT status
# and executing the audio input and transcription process accordingly.
# The transcription results are published as ROS2 String messages to the "gpt_text_input_original" topic.
# Make sure to configure the necessary OpenAI credentials and settings in the provided config file.
#
# Author: Herman Ye

# ROS related
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from openai import OpenAI

from pydub import AudioSegment

# Audio recording related
import sounddevice as sd
from scipy.io.wavfile import write

# GPT related
from gpt_status.gpt_param_server import GPTStatus, GPTStatusOperation
from gpt_status.gpt_config import GPTConfig

# Other libraries
import os

config = GPTConfig()

# openai.organization = config.organization
client = OpenAI(api_key=config.api_key)

class AudioInput(Node):
    def __init__(self):
        super().__init__("audio_input", namespace="gpt")
        # Publisher
        self.publisher = self.create_publisher(
            String, "gpt_text_input_original", 10
        )
        # Timer
        self.create_timer(1, self.run_audio_input_callback)
        
        self.audio_file = "/tmp/gpt_audio.wav"
        self.mp3_audio_file = "/tmp/gpt_audio.mp3"
        # Set the speaker volume to maximum for mini pupper v2
        # If you are using a different robot, please comment out the lines
        self.volume_gain_multiplier = config.volume_gain_multiplier
        self.declare_parameter("mini_pupper", False)  # default is False
        self.is_mini_pupper = self.get_parameter("mini_pupper").value
        if self.is_mini_pupper:
            self.get_logger().info("Mini pupper v2 mode is enabled.")
            os.system("amixer -c 0 sset 'Headphone' 100%")
            self.volume_gain_multiplier = 30
            self.get_logger().info(
                "Volume gain multiplier is set to 30 for mini pupper v2."
            )
        # GPT status initialization
        self.gpt_operation = GPTStatusOperation()
        # Audio input initialization status for console output
        self.get_logger().info("Audio input successfully initialized.")

    def run_audio_input_callback(self):
        gpt_current_status_value = self.gpt_operation.get_gpt_status_value()
        # Check if GPT status is WAITING_USER_INPUT
        if gpt_current_status_value == GPTStatus.WAITING_USER_INPUT.name:
            self.run_audio_input()

    def run_audio_input(self):
        # Recording settings
        duration = config.duration  # Audio recording duration, in seconds
        sample_rate = config.sample_rate  # Sample rate

        # For Mangdang mini pupper v2 quadruped robot, the volume is too low
        # so we need to increase the volume by 30x

        # Step 1: Record audio
        self.get_logger().info("Starting audio recording...")
        audio_data = sd.rec(
            int(duration * sample_rate), samplerate=sample_rate, channels=1
        )
        sd.wait()  # Wait until recording is finished

        # Step 2: Increase the volume by a multiplier
        audio_data *= self.volume_gain_multiplier

        self.get_logger().info("Audio recording complete!")

        # Set GPT status to SPEECH_TO_TEXT_PROCESSING
        self.gpt_operation.set_gpt_status_value(
            GPTStatus.SPEECH_TO_TEXT_PROCESSING.name
        )

        # save as WAV
        write(self.audio_file, config.sample_rate, audio_data)

        # convert wav to mp3 with pydub
        audio = AudioSegment.from_wav(self.audio_file)
        audio.export(self.mp3_audio_file, format='mp3')
        
        with open(self.mp3_audio_file, "rb") as f:
            # audio_bytes = f.read()
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=f
            )

        msg = String()
        msg.data = transcript.text
        self.publisher.publish(msg)
        self.get_logger().info(
            "Audio Input Node publishing: \n'%s'" % msg.data
        )

def main(args=None):
    rclpy.init(args=args)

    audio_input = AudioInput()

    rclpy.spin(audio_input)

    audio_input.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
