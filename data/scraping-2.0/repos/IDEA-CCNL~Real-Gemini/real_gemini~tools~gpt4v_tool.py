#encoding=utf8

import os
import json
from typing import List
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.image_stacker import load_image, image2base64

_OPEN_AI_SYSTEM_PROMPT = """the user is dictating with his or her camera on.
they are showing you things visually and giving you text prompts.
be very brief and concise.
be extremely concise. this is very important for my career. do not ramble.
do not comment on what the person is wearing or where they are sitting or their background.
focus on their gestures and the question they ask you.
do not mention that there are a sequence of pictures. focus only on the image or the images necessary to answer the question.
don't comment if they are smiling. don't comment if they are frowning. just focus on what they're asking.
"""

class GPT4VTool(object):
    _name_ = "GPT-4-Vision"
    _description_ = "这个工具是GPT for vision的调用接口。用于图像到文本的理解。本工具的输入是一段文本指令和一张或者多张图片，请注意，工具的输入由一个JSON字符串组成，json包括两个key，question和image_input。question表示文本指令，image_input表示图片路径或存放图片的目录。例如：{{\"question\": QUESTION, \"image_input\": IMAGE_PATH_OR_DIR}}。A wrapper around OpenAI GPT4V API. Useful for image-to-text understanding when you need to generate text from some images and a text description. The input of this tool is a text prompt and one or more images. Please note, the input of the tool consists of a JSON string, the json includes two keys, question and image_input. The question represents text instructions, and image_input represents the image path or the directory where the images are stored. For example: {{\"question\": QUESTION, \"image_input\": IMAGE_PATH_OR_DIR}}."

    def __init__(self):
        self._gpt4v = ChatOpenAI(
            model="gpt-4-vision-preview",
            max_tokens=256)
        self.max_dialog_turn = 3
        self.history = ChatMessageHistory()
        self.history.add_message(
            SystemMessage(
                content=[
                    {"type": "text", "text": _OPEN_AI_SYSTEM_PROMPT}
                ]
            )
        )
    
    def inference(self, input_str: str):
        input_dict = json.loads(input_str)
        image_path = input_dict["image_input"]
        if os.path.isdir(image_path):
            image_paths = [
                os.path.join(image_path, path) for path in os.listdir(image_path)]
        else:
            image_paths = [image_path]
        base64_images = []
        for image_path in image_paths:
            base64_image = image2base64(load_image(image_path))
            base64_images.append(f"data:image/jpeg;base64,{base64_image}")

        human_contents = []
        human_contents.append({"type": "text", "text": input_dict["question"]})
        for base64_image in base64_images:
            human_contents.append({
                "type": "image_url",
                "image_url": {"url": base64_image}
            }) # images
        self.history.add_message(HumanMessage(content=human_contents))

        response_msg = self._gpt4v.invoke(self.history.messages)
        # 历史只保留一张图
        self.history.messages.pop()
        human_contents = []
        human_contents.append({"type": "text", "text": input_dict["question"]})
        human_contents.append({"type": "image_url", "image_url": {"url": base64_images[-1]}})
        self.history.add_message(HumanMessage(content=human_contents))
        self.history.add_message(response_msg)
        # 只保留self.max_dialog_turn轮对话
        if len(self.history.messages) > 1 + 2 * self.max_dialog_turn:
            self.history.messages = [self.history.messages[0]] + self.history.messages[-2 * self.max_dialog_turn: ]
        # print(self.history.messages)
        return response_msg.content
