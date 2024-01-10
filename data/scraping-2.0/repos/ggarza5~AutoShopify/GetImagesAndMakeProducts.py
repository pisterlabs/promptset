import requests
import json
import time
from openai import OpenAI

# API Configuration
PRINTIFY_API_KEY = "your_printify_api_key"
GPT4_API_KEY = "your_gpt4_api_key"
GPT4_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
SHOP_ID = your_shop_id  # Replace with your shop ID
GPT_MODEL = "gpt-3.5-turbo-1106"

# Generic API URLs
SEARCH_URL = "https://example.com/api/search?q=your_query"
BLUEPRINTS = [50, 1000, 10, 194, 116]
UPLOAD_IMAGE_ENDPOINT = "https://api.example.com/v1/uploads/images.json"
CREATE_PRODUCT_ENDPOINT = f"https://api.example.com/v1/shops/{SHOP_ID}/products.json"

# Rate limiting delay
RATE_LIMIT_DELAY = 0.0125


def fetch_objects():
    response = requests.get(SEARCH_URL, headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        return response.json().get("objectIDs", [])
    else:
        raise Exception("Failed to fetch objects from the API")


def fetch_object_details(object_id):
    time.sleep(RATE_LIMIT_DELAY)
    object_url = f"https://example.com/api/objects/{object_id}"
    response = requests.get(object_url, headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch details for object ID {object_id}")


def get_image_dimensions(object_data):
    measurements = object_data.get("measurements", [])
    for measurement in measurements:
        if measurement.get("elementName") == "Overall":
            return measurement.get("elementMeasurements")
    return None


def fetch_printify_variants(blueprint_id):
    url = f"http://api.printify.com/v1/catalog/blueprints/{blueprint_id}/print_providers/2/variants.json?show-out-of-stock=0"
    response = requests.get(
        url, headers={"Authorization": f"Bearer {PRINTIFY_API_KEY}"}
    )
    if response.status_code == 200:
        variants = response.json().get("variants", [])
        # Add blueprint_id to each variant
        for variant in variants:
            variant["blueprint_id"] = blueprint_id
        return variants
    else:
        raise Exception(f"Failed to fetch variants for blueprint {blueprint_id}")


def calculate_aspect_ratio(dimensions):
    """Calculate the aspect ratio given width and height"""
    if "Width" in dimensions and "Height" in dimensions:
        return dimensions["Width"] / dimensions["Height"]
    return None


def find_best_matching_variant(image_dimensions, variants):
    """Find the variant with the closest aspect ratio to the image"""
    image_aspect_ratio = calculate_aspect_ratio(image_dimensions)

    if image_aspect_ratio is None:
        return None

    closest_variant = None
    smallest_difference = float("inf")

    for variant in variants:
        for placeholder in variant.get("placeholders", []):
            variant_aspect_ratio = calculate_aspect_ratio(
                {"Width": placeholder["width"], "Height": placeholder["height"]}
            )
            if variant_aspect_ratio is not None:
                difference = abs(image_aspect_ratio - variant_aspect_ratio)
                if difference < smallest_difference:
                    smallest_difference = difference
                    closest_variant = variant
    # Set a generic price
    closest_variant["price"] = 100
    return closest_variant


def upload_image_to_printify(image_url):
    payload = {
        "file_name": image_url.split("/")[-1],  # Extracting file name from URL
        "url": image_url,
    }
    response = requests.post(
        UPLOAD_IMAGE_ENDPOINT,
        headers={"Authorization": f"Bearer {PRINTIFY_API_KEY}"},
        json=payload,
    )
    if response.status_code in [200, 201]:
        return response.json().get("id")
    else:
        raise Exception("Failed to upload image")


def generate_product_details(client, object_data, variant_title):
    image_title = object_data.get("title", "Generic Title")
    prompt = f"Create a short product title and description that combines '{image_title}' with the following product variant: '{variant_title}'."

    completion = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    return completion.choices[0].message.content


def create_printify_product(image_id, variant, title, description):
    payload = {
        "title": title,
        "description": description,
        "blueprint_id": variant["blueprint_id"],
        "print_provider_id": 2,  # Assuming a specific print provider
        "variants": [variant],
        "print_areas": [
            {
                "variant_ids": [variant["id"]],
                "placeholders": [
                    {
                        "position": "front",
                        "images": [
                            {"id": image_id, "x": 0.5, "y": 0.5, "scale": 1, "angle": 0}
                        ],
                    }
                ],
            }
        ],
    }
    response = requests.post(
        CREATE_PRODUCT_ENDPOINT,
        headers={"Authorization": f"Bearer {PRINTIFY_API_KEY}"},
        json=payload,
    )
    if response.status_code in [200, 201]:
        return response.json().get("id")
    else:
        raise Exception("Failed to create product")


if __name__ == "__main__":
    object_ids = fetch_objects()

    # Fetch variants for all blueprints
    blueprint_variants = {bp: fetch_printify_variants(bp) for bp in BLUEPRINTS}
    client = OpenAI(api_key=GPT4_API_KEY)

    for object_id in object_ids:
        try:
            object_data = fetch_object_details(object_id)
            image_dimensions = get_image_dimensions(object_data)
            image_url = object_data.get("primaryImage")

            for blueprint_id, variants in blueprint_variants.items():
                best_variant = find_best_matching_variant(image_dimensions, variants)
                if best_variant:
                    variant_title = best_variant.get("title", "Generic Variant")
                    image_id = upload_image_to_printify(image_url)
                    product_details = generate_product_details(
                        client, object_data, variant_title
                    )
                    title, description = product_details.split(
                        "\n", 1
                    )  # Split title and description
                    printify_product_id = create_printify_product(
                        image_id,
                        best_variant,
                        title.strip(),
                        description.strip(),
                    )
                    print("Product Created:", printify_product_id)
                    break  # Break if a suitable variant is found for this object
        except Exception as e:
            print(f"Error processing object ID {object_id}: {str(e)}")
