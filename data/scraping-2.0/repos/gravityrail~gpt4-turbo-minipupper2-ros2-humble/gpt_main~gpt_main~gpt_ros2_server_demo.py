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
# This code defines a ROS2 server node which provides a GPTText service based on the GPT-3.5-turbo or GPT-4 model from OpenAI.
# The GPTText service takes a user input as text prompt, generates chat completion using OpenAI API, and returns a chat response.
# The server node relies on a GPTConfig module to manage LLM model configurations, including handling chat history and API settings.
# A client node (such as the one in the previous example) can communicate with this server node to receive AI-generated responses.
# This example script is useful for integrating a LLM, using the OpenAI API, within a ROS2-based system for natural language processing and chatbot applications. 
#
# Author: Herman Ye

import rclpy
from rclpy.node import Node
from openai import OpenAI

from gpt_interfaces.srv import GPTText
from gpt_status.gpt_config import GPTConfig


config = GPTConfig()
# openai.organization = config.organization
client = OpenAI(api_key=config.api_key)

class GPTServer(Node):
    def __init__(self):
        super().__init__('gpt_ros2_server')
        self.srv = self.create_service(
            GPTText, 'GPT_service', self.gpt_callback)
        self.get_logger().info('GPT Server is ready.')

    def gpt_callback(self, request, response):
        user_prompt = request.request_text
        # user input processor
        input = self.user_input_processor(user_prompt)

        # generate chat completion
        gpt_response = self.generate_chat_completion(input)

        # get chat response text
        output = self.get_chat_response_text(gpt_response)

        # Append user & assitant current messages to chat history
        self.append_message_to_history("user", config.user_prompt)
        self.append_message_to_history("assistant", config.assisstant_response)

        response.response_text = output
        # self.get_logger().info(
        # 'GPT Server has responded: %s' % response.response_text)
        return response

    def user_input_processor(self, user_prompt):
        """
        This function takes the user prompt
        and preprocesses it to be used as input
        """
        input = []
        for message in config.chat_history:
            input.append(
                {"role": message['role'], "content": message['content']})
        config.user_prompt = user_prompt
        input.append({"role": "user", "content": config.user_prompt})
        return input

    def generate_chat_completion(self, input):
        """
        This function takes the input and generates the chat completion
        and returns the response
        """
        response = client.chat.completions.create(model=config.model,
        messages=input,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        stop=config.stop)
        return response

    def get_chat_response_text(self, response):
        """
        This function takes the response
        and returns the chat response text individually
        """
        response_text = response.choices[0].message.content.strip()
        config.assisstant_response = response_text
        return response_text

    def append_message_to_history(self, user_or_ai, content):
        config.chat_history.append({"role": user_or_ai, "content": content})


def main(args=None):
    rclpy.init(args=args)
    gpt_server = GPTServer()
    rclpy.spin(gpt_server)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
