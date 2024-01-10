import openai
import base64
import requests

def analyze_image(api_key, image):
    # Streamlit image에서 제공하는 이미지를 잘 읽을 수 있도록하는 함수.(수정)
    def encode_image(image_file):
        return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Look at this image, analyze the mood, emotion, and behavior in this image, and recommend a music genre. Explain it in less than 200 characters (no double or single quotes)."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        response_json = response.json()
        content_text = response_json.get("choices", [])[0].get("message", {}).get("content", "")
        return content_text
    else:
        return "Error in API request"


# with open("path/to/your/image.jpg", "rb") as image_file:
#     image_data = image_file.read()

# result = analyze_image(api_key, image_data)
# print(result)


