import config
import openai_utils
import replicate_utils
import sinkinai_utils
import getimg_utils

# dummy function for localization
def _(text):
    return text

MODELS = {
    **getimg_utils.MODELS,
    **sinkinai_utils.MODELS,
    **replicate_utils.MODELS,
    "dalle": {
        "name": "DALL·E 3 (OpenAI)",
        "tips": [
            _("The price is for one image"),
            _("Any languages"),
        ],
        "size_options": [
            {
                "width": 1000,
                "height": 1000,
                "cost": config.DALLE_TOKENS,
            },
        ],
    }
}

def calc_cost(model, width, height):
    cost = None
    if model == "dalle":
        cost = config.DALLE_TOKENS
    elif model in MODELS:
        sizes = MODELS[model]["size_options"]
        for size in sizes:
            if width == size["width"] and height == size["height"]:
                cost = size["cost"]
                break
    
    return cost

def _build_image_data(image_urls):
    return [{"image": url} for url in image_urls]

async def inference(model, prompt, width, height):
    result = None
    if model == "dalle":
        images = await openai_utils.create_image(prompt)
        result = _build_image_data([images[0]["url"]])
    elif model in sinkinai_utils.MODELS:
        images = await sinkinai_utils.inference(model=model, width=width, height=height, prompt=prompt)
        result = _build_image_data(images)
    elif model in replicate_utils.MODELS:
        images = await replicate_utils.inference(model=model, width=width, height=height, prompt=prompt)
        result = _build_image_data(images)
    elif model in getimg_utils.MODELS:
        result = await getimg_utils.inference(model=model, width=width, height=height, prompt=prompt)

    if result is None:
        raise Exception("invalid model")
    return result

async def upscale(image, scale: float = 2):
    image = await getimg_utils.upscale(image)
    if image is None:
        raise Exception("failed to upscale")
    return image
