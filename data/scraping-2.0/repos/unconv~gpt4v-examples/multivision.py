from openai import OpenAI
import base64

model = OpenAI()

def image_b64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def look(
    image_paths: list[str] | str,
    prompt="Describe this image",
):
    if not isinstance(image_paths, list):
        image_paths = [image_paths]

    images = []

    for image in image_paths:
        b64_image = image_b64(image)
        images.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{b64_image}",
        })

    response = model.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": images + [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ],
        max_tokens=1024,
    )

    message = response.choices[0].message
    return message.content
