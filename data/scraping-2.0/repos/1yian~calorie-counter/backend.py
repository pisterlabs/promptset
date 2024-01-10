import openai
import base64
from PIL import Image
import io
import re

DEFAULT_PROMPT = "This is a picture of a meal. To the best of your ability, come up with a maximum " + \
"likelihood estimate of the macro and calorie count of this meal based on how " + \
"it looks alone. When you come up with your estimate, call the estimate " + \
"function by saying 'estimate(x, p, f, c)' where x is your best guess of " + \
"the calories, p is protein, f is fats, and c is carbs. For example, estimate(10, 1, 3, 4)"

DEFAULT_PROMPT_NOTES = "This is a picture of a meal. To the best of your ability, come up with a maximum " + \
"likelihood estimate of the macro and calorie count of this meal based on how " + \
"it looks alone. Here are some notes about the meal: [NOTES]." + \
"When you come up with your estimate, call the estimate " + \
"function by saying 'estimate(x, p, f, c)' where x is your best guess of " + \
"the calories, p is protein, f is fats, and c is carbs. For example, estimate(10, 1, 3, 4)"

class OpenAIVisionAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def find_function_call(self, text):
        # Regex pattern to match the function call with four decimal numbers
        # \d+ matches one or more digits, \.?\d* matches an optional decimal point followed by zero or more digits
        pattern = r"estimate\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)"

        # Find all matches in the text
        matches = re.findall(pattern, text)

        # Process matches
        function_calls = []
        for match in matches:
            # Convert string numbers to floats
            numbers = [float(num) for num in match]
            function_calls.append(numbers)

        return function_calls

    def create_prompt(self, notes: str):
        if notes:
            return DEFAULT_PROMPT_NOTES.replace("[NOTES]", notes)
        else:
            return DEFAULT_PROMPT


    def query_vision_model(self, image, notes, count=0):
        if count > 3:
            assert False, "NOT WORKING"
        text_input = self.create_prompt(notes)
        base64_image = self.encode_image(image)
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_input},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=900,
            temperature=1.3,
        )
        answer = response.choices[0].message['content']

        macros_found = self.find_function_call(answer)
        if len(macros_found) > 0:
            calories, fats, proteins, carbs = macros_found[0]
        else:
            print("Count:", count, "Calling again, AI output was", answer)
            return self.query_vision_model(image, notes, count=count+1)
            calories, fats, proteins, carbs = -1, -1, -1, -1
        return calories, fats, proteins, carbs
