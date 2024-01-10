
from PIL import Image
import pytesseract
from transformers import AutoModelForImageClassification, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score


import base64
import json
import requests
from PIL import Image
import torch
import openai
class VisionFunctions:
    def __init__(self, api_base_url: str = None):
        self.api_base_url = api_base_url  # Base URL where the Vision model API is hosted
        if self.api_base_url is None:
            # Initialize your image model and tokenizer here if not using API
            self.model = YourImageModel.from_pretrained('your-model-name')
            self.tokenizer = YourImageTokenizer.from_pretrained('your-tokenizer-name')

    def image_analysis(self, url, image_path):
        if self.api_base_url:
            # API-based approach
            with open(image_path, 'rb') as f:
                img_str = base64.b64encode(f.read()).decode('utf-8')
            
            prompt = f'<img src="data:image/jpeg;base64,{img_str}">'
            response = requests.post(
                'http://127.0.0.1:5000/api/v1/generate',
                json={'prompt': prompt, 'stopping_strings': ['\n###']}
            )
            
            if response.status_code == 200:
                prediction = json.loads(response.text)
                return prediction['results'][0]['text']
            else:
                print(f"Error: {response.status_code}")
                return None
        else:
            # Direct model inference approach
            image = Image.open(image_path).convert('RGB')
            inputs = self.tokenizer(image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_category = self.model.config.id2label[torch.argmax(logits).item()]
            return predicted_category

    def code_extraction(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        extracted_code = pytesseract.image_to_string(image)
        return extracted_code

    def self_improvement(self) -> Dict[str, float]:
        """
        Use metrics like accuracy to fine-tune the vision algorithms.
        This function returns a dictionary containing these metrics for evaluation.
        """
        all_predicted_categories = list(self.predicted_categories.values())
        all_true_categories = list(self.true_categories.values())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_true_categories, all_predicted_categories)
        
        metrics = {
            'Accuracy': accuracy
        }
        
        # These metrics can then be used to fine-tune the vision algorithms
        # For demonstration, we're just returning these metrics
        return metrics


