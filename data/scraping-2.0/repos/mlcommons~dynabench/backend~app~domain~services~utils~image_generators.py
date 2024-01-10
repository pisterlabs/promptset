# Copyright (c) MLCommons and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import base64
import os
from abc import ABC, abstractmethod
from io import BytesIO

import openai
import requests
from requests.adapters import HTTPAdapter

from app.domain.services.base.task import TaskService
from app.domain.services.utils.adapters import RequestSession
from app.domain.services.utils.constant import forbidden_image


class ImageProvider(ABC):
    def __init__(self):
        self.task_service = TaskService()
        pass

    @abstractmethod
    def generate_images(self, prompt: str, num_images: int) -> list:
        pass

    @property
    @abstractmethod
    def provider_name(self):
        pass


class OpenAIImageProvider(ImageProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI")
        openai.api_key = self.api_key

    def generate_images(
        self, prompt: str, num_images: int, models: list = [], endpoint: str = ""
    ) -> list:
        openai.api_key = self.api_key
        try:
            response = openai.Image.create(
                prompt=prompt, n=1, size="256x256", response_format="b64_json"
            )
            image_response = [x["b64_json"] for x in response["data"]]
            return {"generator": self.provider_name(), "images": image_response}

        except Exception as e:
            print(e, "This was the exception")
            images = [forbidden_image]
            return {"generator": self.provider_name(), "images": images}

    def provider_name(self):
        return "openai"


class DynabenchImageProvider(ImageProvider):
    def __init__(self):
        pass

    def generate_images(self, prompt: str, num_images: int, model, endpoint) -> list:
        payload = {"prompt": prompt, "num_images": 1}
        response = requests.post(f"{endpoint['dynabench']['endpoint']}", json=payload)
        try:
            if response.status_code == 200:
                return {"generator": self.provider_name(), "images": response.json()}
        except:
            return {"generator": self.provider_name(), "images": [forbidden_image]}

    def provider_name(self):
        return "dynabench"


class SD2ImageProvider(ImageProvider):
    def __init__(self):
        self.api_key = os.getenv("HF")
        self.session = RequestSession()

    def generate_images(self, prompt: str, num_images: int, model, endpoint) -> list:
        print("Trying model", endpoint["hf_inference_3"]["endpoint"])
        payload = {"inputs": prompt, "steps": 30}
        headers = {"Authorization": self.api_key}
        try:
            response = self.session.session.post(
                f"{endpoint['hf_inference_3']['endpoint']}",
                json=payload,
                headers=headers,
            )
            if response.status_code == 200:
                new_image = response.json()
                return {"generator": self.provider_name(), "images": [new_image]}
        except:
            return {"generator": self.provider_name(), "images": [forbidden_image]}

    def provider_name(self):
        return "sd2.1-base"


class SDXLImageProvider(ImageProvider):
    def __init__(self):
        self.api_key = os.getenv("HF")
        self.session = RequestSession()

    def generate_images(self, prompt: str, num_images: int, model, endpoint) -> list:
        print("Trying model", endpoint["hf_inference_4"]["endpoint"])
        payload = {"inputs": prompt, "steps": 30}
        headers = {"Authorization": self.api_key}
        try:
            response = self.session.session.post(
                f"{endpoint['hf_inference_4']['endpoint']}",
                json=payload,
                headers=headers,
                timeout=25,
            )
            if response.status_code == 200:
                new_image = response.json()[0]["image"]["images"][0]
                return {"generator": self.provider_name(), "images": [new_image]}
        except:
            return {"generator": self.provider_name(), "images": [forbidden_image]}

    def provider_name(self):
        return "sdxl1.0"


class SDRunwayMLImageProvider(ImageProvider):
    def __init__(self):
        self.api_key = os.getenv("HF")
        self.session = RequestSession()

    def generate_images(self, prompt: str, num_images: int, model, endpoint) -> list:
        print("Trying model", endpoint["hf_inference_2"]["endpoint"])
        payload = {"inputs": prompt, "steps": 30}
        headers = {"Authorization": self.api_key}
        try:
            response = self.session.session.post(
                f"{endpoint['hf_inference_2']['endpoint']}",
                json=payload,
                headers=headers,
            )
            if response.status_code == 200:
                new_image = response.json()
                return {"generator": self.provider_name(), "images": [new_image]}
        except:
            return {"generator": self.provider_name(), "images": [forbidden_image]}

    def provider_name(self):
        return "runwayml-sd1.5"


class SDVariableAutoEncoder(ImageProvider):
    def __init__(self):
        self.api_key = os.getenv("HF")
        self.session = RequestSession()

    def generate_images(self, prompt: str, num_images: int, model, endpoint) -> list:
        print("Trying model", endpoint["hf_inference_1"]["endpoint"])
        payload = {"inputs": prompt, "steps": 30}
        headers = {"Authorization": self.api_key}
        try:
            response = self.session.session.post(
                f"{endpoint['hf_inference_1']['endpoint']}",
                json=payload,
                headers=headers,
            )
            if response.status_code == 200:
                new_image = response.json()[0]["image"]["images"][0]
                return {"generator": self.provider_name(), "images": [new_image]}
        except:
            return {"generator": self.provider_name(), "images": [forbidden_image]}

    def provider_name(self):
        return "sd+vae_ft_mse"


class HFInferenceEndpointImageProvider(ImageProvider):
    def __init__(self):
        self.api_key = os.getenv("HF")

    def generate_images(self, prompt: str, num_images: int, model, endpoint) -> list:
        print("Trying model", endpoint["hf_inference_1"]["endpoint"])
        payload = {"inputs": prompt, "steps": 30}
        headers = {"Authorization": self.api_key}

        response = requests.post(
            f"{endpoint['hf_inference_1']['endpoint']}", json=payload, headers=headers
        )
        if response.status_code == 200:
            new_image = response.json()[0]["image"]["images"][0]
            return {"generator": self.provider_name(), "images": [new_image]}

    def provider_name(self):
        return "sd+vae_ft_mse"


class HFImageProvider(ImageProvider):
    def __init__(self):
        self.api_key = os.getenv("HF")

    def generate_images(
        self, prompt: str, num_images: int, models: list, endpoint: str
    ):
        payload = {"inputs": prompt}
        headers = {
            "Authorization": self.api_key,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "x-use-cache": "false",
        }
        model = models["huggingface"]["models"][0]
        endpoint = f"{endpoint['huggingface']['endpoint']}/{model}"
        images = []
        try:
            response = requests.post(endpoint, json=payload, headers=headers)
            print("Trying model", endpoint, "with status code", response.status_code)
            if response.status_code == 200:
                base64_image = base64.b64encode(response.content)
                images.append(base64_image.decode("utf-8"))
        except response.exceptions.Timeout:
            print("Timeout error")
        return {"generator": self.provider_name(), "images": images}

    def provider_name(self):
        return "hf"


class StableDiffusionImageProvider(ImageProvider):
    def __init__(self):
        self.api_key = os.getenv("STABLE_DIFFUSION")

    def generate_images(
        self, prompt: str, num_images: int, models: list, endpoint: str
    ) -> list:
        for model in models["together"]["models"]:
            print(f"Trying model {model}")
            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "prompt": prompt,
                    "n": num_images,
                    "steps": 20,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "",
                },
            )
            try:
                if res.status_code == 200:
                    image_response = res.json().get("output").get("choices")
                    image_response = [x["image_base64"] for x in image_response]
                    print(f"Model {model} worked")
                    return image_response
            except requests.exceptions.Timeout:
                continue
            else:
                continue

        image_response = [forbidden_image] * int(num_images)
        return image_response

    def provider_name(self):
        return "stable_diffusion"


class MidjourneyImageProvider(ImageProvider):
    def __init__(self):
        self.server_id = os.getenv("SERVER_ID")
        self.channel_id = os.getenv("CHANNEL_ID")
        self.dyna_bot = os.getenv("DYNA_BOT")

    def generate_images(
        self, prompt: str, num_images: int, models: list, endpoint: str
    ) -> list:
        payload = {
            "type": 2,
            "application_id": "1109849712405782700",
            "guild_id": self.server_id,
            "channel_id": self.channel_id,
            "session_id": "2fb980f65e5c9a77c96ca01f2c242cf6",
            "data": {
                "version": "1077969938624553050",
                "id": "938956540159881230",
                "name": "imagine",
                "type": 1,
                "options": [{"type": 3, "name": "prompt", "value": prompt}],
                "application_command": {
                    "id": "938956540159881230",
                    "application_id": "1109849712405782700",
                    "version": "1077969938624553050",
                    "default_permission": True,
                    "default_member_permissions": None,
                    "type": 1,
                    "nsfw": False,
                    "name": "imagine",
                    "description": "Create images with Midjourney",
                    "dm_permission": True,
                    "options": [
                        {
                            "type": 3,
                            "name": "prompt",
                            "description": "The prompt to imagine",
                            "required": True,
                        }
                    ],
                },
                "attachments": [],
            },
        }

        header = {"authorization": self.dyna_bot}
        response = requests.post(
            "https://discord.com/api/v9/interactions", json=payload, headers=header
        )

        print(response)
        return

    def provider_name(self):
        return "midjourney"
