from openai import OpenAI
import base64
import json
import os


class OpenAIClient:
    def __init__(self, default_language=None):
        self.client = OpenAI()
        self.default_model = 'gpt-4-1106-preview'
        self.vision_model = 'gpt-4-vision-preview'
        self.image_generation_model = 'dall-e-3'
        if not default_language:
            self.default_language = "French"
        else:
            self.default_language = default_language
        print(f"Default language: {self.default_language}")

    @staticmethod
    def decode_response(response):
        return response.choices[0].message.content

    @staticmethod
    def decode_json_response(response):
        decoded_response = OpenAIClient.decode_response(response)
        return json.loads(decoded_response)

    def plant_checker(self, plant_name: str):
        """
        Given a plant name, will call openai to generate a response on how to take care of the plant.
        The response will be in JSON format and compliant with the pyPlants.models.Plant model.
        """
        print(f"Default language: {self.default_language}")
        response = self.client.chat.completions.create(
            model=self.default_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You're a plant expert, designed to help users to take care of their plants."
                               "\nFor each given plant, you should always provide the following information in the JSON format:"
                               f"\n- description: A description of the plant in {self.default_language} (string)"
                               "\n- water_frequency_summer: Summer Watering frequency, in days (integer)"
                               "\n- water_frequency_winter: Winter Watering frequency, in days (integer)"
                               "\n- sunlight: How much sunlight it needs (light_exposure, partial_shade or shade)"
                               "\n- sun_exposure: What type of sun exposure (direct_sun or no_direct_sun)"
                               "\n- fertilizer: If it needs to be fertilized (True or False)"
                               "\n- fertilizer_season: The fertilizing season (spring, summer, fall or winter, 1 choice only) if it needs to be fertilized"
                               "\n- repotting: If it needs to be repotted (True or False)"
                               "\n- repotting_season: The repotting season (spring, summer, fall or winter, 1 choice only) if it needs to be repotted"
                               "\n- leaf_mist: If its leaves need to be misted (True or False)"
                               f"\n- extra_tips: Provide any additional tips for taking care of the plant (string) - in {self.default_language}, for non plant experts."
                               "\n In case you don't know the plant, instead you should answer with a json: 'error: unknown plant''"
                },
                {
                    "role": "user",
                    "content": f"How should I take care of my {plant_name}?"
                }
            ]
        )
        return response

    def plant_recognizer(self, image_path: str):
        """
        Given an image of a plant, will call OpenAI to generate a response on what plant it is.
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Error: The file at {image_path} does not exist.")

        file_size = os.path.getsize(image_path)
        print(f"File size: {file_size} bytes")
        if file_size >= 20 * 1024 * 1024:  # 20 MB
            raise ValueError("Error: File size exceeds 20 MB limit.")

        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error reading or encoding the image: {e}")
            raise e

        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You're a plant expert, designed to help users to take care of their plants. "
                                   f"You can recognize plants provided by the user and ONLY answer with the common name "
                                   f"of the plant (in {self.default_language})."
                                   "If you can't recognize the plant, you should only answer with 'unknown'"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is the name of this plant?"
                            },
                            {
                                "type": "image",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_string}"
                                }
                            }
                        ]
                    }
                ]
            )
            return response
        except Exception as e:
            print(f"Error making request to OpenAI: {e}")
            raise e

    def plant_image_generator(self, plant_name: str):
        """
        Given a plant name, will call openai to generate an image of the plant.
        """
        response = self.client.images.generate(
            model=self.image_generation_model,
            prompt=f"A vibrant illustration of a {plant_name} plant, with a smooth, shiny finish and bright, popping colors. "
                   f"The plant should be centered within a circular badge that glows with a radiant light effect, giving a sense of premium quality. "
                   f"The background should feature a light gradient that complements the colors of the plant. "
                   f"The artwork should be detailed with a clear outline and a slight drop shadow for depth, resembling a high-quality sticker design.",
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response


