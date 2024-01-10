from dotenv import load_dotenv
from transcriptassist.ocr.google_cloud_vision import GoogleCloudVision
from transcriptassist.ocr.mathpix import MathPix
import os
import json
import openai
from transcriptassist.utils import process_image, generate_prompt


class TranscriptAssist:
    def __init__(self, gpt_temperature=0.3, previous_messages_json="transcriptassist/gpt/history.json",
                 temp_dir="transcriptassist/temp/"):

        self._get_credentials()
        self.GoogleOCR = GoogleCloudVision(self.gcv_API_KEY, self.gcv_PROJECT_ID)
        self.MathPixOCR = MathPix(self.mathpix_APP_KEY, self.mathpix_APP_ID)
        self.previous_messages = json.load(open(previous_messages_json, "r"))
        openai.api_key = self.openai_API_KEY

        self.gpt_temperature = gpt_temperature

        self.temp_dir = temp_dir


    def get_ocr_transcriptions(self, path, use_mathpix=True, use_gcv=True):
        gcv_text = self.GoogleOCR.get_all_text(path) if use_gcv else None
        mathpix_text = self.MathPixOCR.get_all_text(path) if use_mathpix else None
        return gcv_text, mathpix_text

    def generate_gpt_corrected_transcription(self, prompt, input_temp=None):
        if not prompt:
            return "OCRs failed to detect any text."
        temperature = input_temp if input_temp else self.gpt_temperature
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.previous_messages + [{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=temperature)
        return response.choices[0]['message']['content']

    def transcribe(self, link_or_path, crop_coordinates=None, show_image=False, input_temp=None):
        image_data = process_image(link_or_path, crop_coordinates, show_image)

        temp_file_path = self.temp_dir + "temp_image.png"
        os.makedirs(self.temp_dir, exist_ok=True)
        with open(temp_file_path, 'wb') as f:
            f.write(image_data)

        gcv_text, mathpix_text = self.get_ocr_transcriptions(temp_file_path)
        os.remove(temp_file_path), os.rmdir(self.temp_dir)

        prompt = generate_prompt(gcv_text, mathpix_text)
        corrected_transcription = self.generate_gpt_corrected_transcription(prompt, input_temp)
        return corrected_transcription

    def _get_credentials(self):
        load_dotenv()
        self.gcv_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
        self.gcv_PROJECT_ID = os.getenv("GOOGLE_CLOUD_APP_ID")
        self.mathpix_APP_KEY = os.getenv("MATHPIX_APP_KEY")
        self.mathpix_APP_ID = os.getenv("MATHPIX_APP_ID")
        self.openai_API_KEY = os.getenv("OPENAI_API_KEY")



