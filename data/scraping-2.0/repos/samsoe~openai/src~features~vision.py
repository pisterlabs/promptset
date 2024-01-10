from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()


def get_image_description(image_url):
    # Define the client
    client = OpenAI()

    # Make the API call, passing the image URL as a parameter
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Classify the image by species, common_name, num_animals, date, location, and other metadata in CSV format.",
                    },
                    {"type": "image_url", "image_url": image_url},
                ],
            }
        ],
        max_tokens=300,
    )

    # Return the response
    return response.choices[0]


# Example usage:
image_url = "https://ranch-buckeye.s3.amazonaws.com/OrchardHouse/Cam3/p_014016.jpg"
description = get_image_description(image_url)

# Display the result
print(description.message.content)
