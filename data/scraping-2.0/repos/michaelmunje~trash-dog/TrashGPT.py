from openai import OpenAI
import requests
import os


class Trash:
    """
    Represents a detected trash object in an image.
    
    Attributes:
        name (str): The name of the trash object.
        location (list of int): The location of the object in the image, given as [y, x] coordinates.
    """
    def __init__(self, name: str, location: [int, int]):
        self.name = name
        self.location = location # [y, x]


class TrashGPT:
    def __init__(self):
        self.client = OpenAI()

        self.trash_prompt = "Is there any trash in this image?" + \
            " Provide the response as one trash item " + \
            " and nothing else, formatted as 'Object Name: Relative Center Coordinates (Y, X)'." + \
            "\nExample: 'Soda can: 0.35, 0.58'."

        self.trash_can_prompt = "Is there a trash can in this image?" + \
            " Provide the response as one trash item " + \
            " and nothing else, formatted as 'Object Name: Relative Center Coordinates (Y, X)'." + \
            "\nExample: 'Soda can: 0.35, 0.58'."

        api_key = os.environ.get('OPENAI_API_KEY')
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        del api_key
        self.model_name = "gpt-4-vision-preview"
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def get_payload(self, image, prompt=None):
        if prompt is None:
            prompt = self.trash_prompt

        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                                }
                            }
                        ]
                    }
                ],
            "max_tokens": 300
        }

    def perform_detection(self, image, image_height, image_width, n_runs=1, prompt=None) -> [Trash]:
        runs = []
        for _ in range(n_runs):
            if prompt is None:
                prompt = self.trash_prompt
            payload = self.get_payload(image, prompt=prompt)
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            output = response.json()
            gpt_output =  output['choices'][0]['message']['content']
            try:
                trash = None
                contents = gpt_output.split(':')
                trash_name = contents[0]
                trash_locations = contents[-1].split(',')
                trash_y = int(float(trash_locations[0]) * image_height)
                trash_x = int(float(trash_locations[1]) * image_width)
                trash = Trash(trash_name, [trash_y, trash_x])
            except:
                trash = None
            runs.append(trash)

        self.raw_runs = runs
        # let's get avg of runs
        n_successful_runs = 0
        total_y, total_x = 0, 0
        trash_name = 'None'
        for i in range(n_runs):
            if runs[i] is not None:
                total_y += runs[i].location[0]
                total_x += runs[i].location[1]
                n_successful_runs += 1
                trash_name = runs[i].name
        
        return Trash(trash_name, [total_y // n_successful_runs, total_x // n_successful_runs])
    
    def perform_trash_detection(self, image, image_height, image_width, n_runs=1) -> [Trash]:
        return self.perform_detection(image, image_height, image_width, n_runs=n_runs, prompt=self.trash_prompt)
    
    def perform_trash_can_detection(self, image, image_height, image_width, n_runs=1) -> [Trash]:
        return self.perform_detection(image, image_height, image_width, n_runs=n_runs, prompt=self.trash_can_prompt)
    
    def overlay_results(self, image, trash: Trash):
        """
        Overlays red dots on the image at the given locations.
        """
        image = image.copy()
        location = trash.location
        image[location[0] - 3:location[0] + 3, location[1] - 3:location[1] + 3] = [0, 0, 255]
        return image
