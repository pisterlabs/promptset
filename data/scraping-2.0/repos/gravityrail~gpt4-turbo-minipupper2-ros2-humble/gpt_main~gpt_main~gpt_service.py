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
# This script creates a ROS2 node named "gpt_service" that listens to a topic for text input,
# processes it using the OpenAI GPT-3 API, and publishes the generated text response.
# When receiving a text message from the user, the node preprocesses it, generates a response,
# and appends the user's input and GPT-3 assistant's response to the chat history.
# The GPTService class includes a callback to handle the text input and methods to handle
# general input processing, generating chat completions, and creating response text.
# The main function initializes and runs the GPTService node.
#
# Author: Herman Ye

# ROS related
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# GPT related
from openai import OpenAI

# GPT status related
from gpt_status.gpt_param_server import GPTStatus, GPTStatusOperation

# GPT config related
from gpt_status.gpt_config import GPTConfig

config = GPTConfig()
# openai.organization = config.organization

client = OpenAI(api_key=config.api_key)

class GPTService(Node):
    def __init__(self):
        super().__init__("gpt_service", namespace="gpt")
        self.subscription = self.create_subscription(
            String, "gpt_text_input_original", self.gpt_callback, 10
        )
        self.publisher = self.create_publisher(String, "gpt_text_output", 10)
        self.get_logger().info("GPT node is ready.")

        # GPT status initialization
        self.gpt_operation = GPTStatusOperation()

    def gpt_callback(self, msg):
        self.get_logger().info("GPT node has received: %s" % msg.data)
        # Set GPT status to GPT_PROCESSING
        self.gpt_operation.set_gpt_status_value(GPTStatus.GPT_PROCESSING.name)
        user_prompt = msg.data
        self.get_logger().info("GPT node is processing: %s" % user_prompt)
        # User input processor
        input = self.user_input_processor(user_prompt)
        self.get_logger().info("user_input_processor has finished.")
        # Generate chat completion
        gpt_response = self.generate_chat_completion(input)
        self.get_logger().info("generate_chat_completion has finished.")
        # Get chat response text
        output = self.get_chat_response_text(gpt_response)
        self.get_logger().info("get_chat_response_text has finished.")
        # Append user & assistant messages to the chat history
        self.append_message_to_history("user", config.user_prompt)
        self.append_message_to_history("assistant", config.assistant_response)

        # Publish the response to the /gpt_text_output topic
        response_msg = String(data=output)
        self.publisher.publish(response_msg)
        self.get_logger().info(
            "GPT service node has published: %s" % response_msg
        )

    def user_input_processor(self, user_prompt):
        """
        This function takes the user prompt
        and preprocesses it to be used as input
        """
        input = []
        for message in config.chat_history:
            input.append(
                {"role": message["role"], "content": message["content"]}
            )
        config.user_prompt = user_prompt
        input.append({"role": "user", "content": config.user_prompt})
        return input

    def generate_chat_completion(self, input):
        """
        This function takes the input and generates the chat completion
        and returns the response
        """
        try:
            response = client.chat.completions.create(model=config.model,
            messages=input,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            stop=config.stop)
        except Exception as e:
            # Handle the error as per your requirement
            self.get_logger().error(f"Error: {e}")
            response = None
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
    gpt_service = GPTService()
    rclpy.spin(gpt_service)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
