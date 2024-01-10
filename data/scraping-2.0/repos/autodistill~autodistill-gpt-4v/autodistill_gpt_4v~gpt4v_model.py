import base64
import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from openai import OpenAI

HOME = os.path.expanduser("~")

@dataclass
class GPT4V(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology, api_key, prompt: str = None):
        self.client = OpenAI(api_key=api_key)
        self.ontology = ontology
        self.prompt = prompt

    def set_of_marks(self, input, masked_input, classes, masks) -> sv.Detections:
        if classes is None:
            classes = {k:k for k in self.ontology.prompts()}

        payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Attached is an image and a set of selections for this image. Please return the number of any selections for a {', '.join(classes)}. Return each class on a new line like Banana: 1, 2, 3 [new line] Apple: 5, 7, 8."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,"
                            + base64.b64encode(open(input, "rb").read()).decode(
                                "utf-8"
                            ),
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,"
                            + base64.b64encode(open(masked_input, "rb").read()).decode(
                                "utf-8"
                            ),
                        }
                    },
                ],
            }
        ]

        print(payload[0]["content"][0])

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=payload,
            max_tokens=300,
        )

        print(response)

        if "none" not in self.ontology.prompts():
            self.ontology.prompts().append("none")

        class_map = {}
        
        for line in response.choices[0].message.content.split("\n"):
            if ":" in line:
                class_name = line.split(":")[0].strip().lower()

                if class_name not in self.ontology.prompts():
                    continue
                
                if class_name not in class_map:
                    class_map[class_name] = []
                
                try:
                    int(line.split(":")[1].strip().lower())
                except:
                    continue

                class_map[class_name].append(int(line.split(":")[1].strip().lower()))

        # get ids from class_map
        all_ids = [item for sublist in class_map.values() for item in sublist]

        print(all_ids)

        # change class id for each mask
        for idx, _ in enumerate(masks.xyxy):
            for class_name, mask_ids in class_map.items():
                for mask_id in mask_ids:
                    if mask_id == idx:
                        masks.class_id[idx] = self.ontology.prompts().index(class_name)
                        break

        new_masks = []
        new_class_id = []
        new_xyxy = []

        for idx, item in enumerate(masks.class_id):
            if item in all_ids:
                new_masks.append(masks.mask[idx])
                new_class_id.append(masks.class_id[idx])
                new_xyxy.append(masks.xyxy[idx])

        masks.confidence = np.array([1] * len(masks.class_id))

        return sv.Detections(
            xyxy=np.array(new_xyxy),
            mask=np.array(new_masks),
            class_id=np.array(new_class_id),
            confidence=np.array([1] * len(new_masks))
        )

    def predict(self, input, classes: list = None) -> sv.Classifications:
        if classes is None:
            classes = {k:k for k in self.ontology.prompts()}

        payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"What is in the image? Return the class of the object in the image. Here are the classes: {', '.join(classes)}. You can only return one class from that list." if self.prompt is None else self.prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,"
                            + base64.b64encode(open(input, "rb").read()).decode(
                                "utf-8"
                            ),
                        }
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=payload,
            max_tokens=300,
        )

        if "none" not in self.ontology.prompts():
            self.ontology.prompts().append("none")

        if self.prompt:
            class_ids = []
            
            for line in response.choices[0].message.content.split("\n"):
                if ":" in line:
                    class_name = line.split(":")[1].strip().lower()
                    print(class_name)
                    if class_name not in self.ontology.prompts():
                        class_ids.append(self.ontology.prompts().index("none"))
                    else:
                        class_ids.append(self.ontology.prompts().index(class_name))
                    
                    print(line)

            confidence = [1] * len(class_ids)
        else:
            result = response.choices[0].message.content.lower()
            if result not in self.ontology.prompts():
                class_ids.append(self.ontology.prompts().index("none"))
                confidence = [0]
            else:
                class_ids = [self.ontology.prompts().index(result)]
                confidence = [1]
        
        return sv.Classifications(
            class_id=np.array(class_ids),
            confidence=np.array(confidence),
        )
