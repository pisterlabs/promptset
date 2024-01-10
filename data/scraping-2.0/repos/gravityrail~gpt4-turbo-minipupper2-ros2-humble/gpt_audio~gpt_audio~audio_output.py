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
# This Python file is a ROS node that uses the OpenAI text to speech service to synthesize speech from text.
# The node subscribes to a topic named gpt_text_output and, when it receives a message on that topic,
# it calls the TTS service to synthesize speech from the message's text.
# The synthesized speech is then saved to a file named /tmp/speech_output.mp3 and played using the mpv command.
# The node also sets the GPT status to ROBOT_ACTION when the speech is finished.
#
# Test method: ros2 topic pub /gpt/gpt_text_output std_msgs/msg/String "{data: 'bark bark bark, beep beep beep'}" -1
#
# Author: Herman Ye

# ROS related
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# GPT related
from gpt_status.gpt_param_server import GPTStatus, GPTStatusOperation
from gpt_status.gpt_config import GPTConfig
from openai import OpenAI

# Other libraries
import os

config = GPTConfig()
client = OpenAI(api_key=config.api_key)


class AudioOutput(Node):
    def __init__(self):
        super().__init__("audio_output", namespace="gpt")
        self.subscription = self.create_subscription(
            String, "gpt_text_output", self.text_callback, 10
        )
        self.get_logger().info("Text to speech node successfully initialized.")
        self.get_logger().info("Waiting for text to speech input...")

        # GPT status initialization
        self.gpt_operation = GPTStatusOperation()

        self.declare_parameter("mini_pupper", False)  # default is False
        self.is_mini_pupper = self.get_parameter("mini_pupper").value

    def text_callback(self, msg):
        self.get_logger().info("Received text: '%s'" % msg.data)
        self.gpt_operation.set_gpt_status_value(
            GPTStatus.TEXT_TO_SPEECH_PROCESSING.name
        )
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=msg.data,
        )
        # Set GPT status to ROBOT_ACTION
        self.gpt_operation.set_gpt_status_value(GPTStatus.ROBOT_ACTION.name)
        # Save the audio output to a file
        output_file_path = "/tmp/speech_output.mp3"
        response.stream_to_file(output_file_path)
        if self.is_mini_pupper:
            # os.system("mpv --audio-device=alsa/hw:1,0" + " " + output_file_path)
            os.system("mpv" + " " + output_file_path)
        else:
            os.system("mpv" + " " + output_file_path)
        self.get_logger().info("Finished OpenAI TTS playing.")

        # If you want to set GPT status to WAITING_USER_INPUT without ROBOT ACTION, uncomment the following line
        # self.gpt_operation.set_gpt_status_value(GPTStatus.WAITING_USER_INPUT.name)


def main(args=None):
    rclpy.init(args=args)

    audio_output = AudioOutput()

    rclpy.spin(audio_output)

    audio_output.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
