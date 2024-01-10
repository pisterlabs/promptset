from openai import OpenAI
import base64
import json


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def find_json_array(text):
    start_index = text.find("[")
    end_index = text.rfind("]")

    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_string = text[start_index:end_index+1]
        try:
            # Convert the extracted string to a JSON object
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            return None
    else:
        print("JSON array not found")
        return None

def call_gpt(image_path, event_info):
    base64_image = encode_image(image_path)
    system_prompt = f"""
        You are given an image with numbers. I want to organize the following event: {str(event_info)}. Output a schedule for the event given the rooms. The schedule should include the start and end time for each event, as well as the name of the event. The schedule should be in json format at the end.
        [
            {{
            day: "",
            activities: [
                {{
                name: "",
                start_time: "",
                end_time: "",
                room: ""
                }}
            ]
            }}
        ]
        """
    client = OpenAI(api_key="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")
    vision_model = "gpt-4-vision-preview"
    response = client.chat.completions.create(
    model=vision_model,
    
    messages=[
        {
            "role": 'system',
            "content": system_prompt,
        },
        {
        "role": "user",
        "content": [
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            },
            },
            {
                "type": 'text',
                "text": 'Output a schedule for the event given the rooms.',
            },
        ],
        }
    ],
    max_tokens=1000,
    )

    response_content = response.choices[0].message.content
    print(response_content)
    extracted_json = find_json_array(response_content)
    return extracted_json


