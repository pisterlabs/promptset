from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import to_pil_image
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import openai
openai.api_key = ''
import base64
import io
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

class AnonymizationModel:
    def __init__(self) -> None:
        
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).cuda()
        self.model = self.model.eval()
        self.model = self.model.to(device)

        self.preproccess = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
        
    def get_mask(self, image):
        input_tensor = self.preproccess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        mask = output_predictions != 15
        mask_rgba = np.zeros((input_tensor.shape[1], input_tensor.shape[2],4))
        mask_rgba[:,:,:3] = np.uint8(image)
        mask_rgba[:,:,3] = np.uint8(mask.cpu().numpy()*255)

        # Save mask
        im_a = Image.fromarray(np.uint8(mask.cpu().numpy()*255), 'L')
        im_rgba = image.copy()
        im_rgba.putalpha(im_a)
        return im_rgba

    def inpainting(self, original_path, mask_path):
        response = openai.Image.create_edit(
        image=open(original_path, "rb"),
        mask=open(mask_path, "rb"),
        prompt="A photo of a hotel room without people",
        n=1,
        size="512x512",
        response_format='b64_json'

        )
        b64_image = response["data"][0]["b64_json"]

        # Decode the base64 image
        image_bytes = base64.b64decode(b64_image)

        # Create a BytesIO object and read the image bytes
        image_buf = io.BytesIO(image_bytes)
        image = Image.open(image_buf)

        # return the image
        return image