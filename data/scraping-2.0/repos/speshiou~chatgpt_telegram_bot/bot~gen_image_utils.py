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
        "name": "DALL·E (OpenAI)",
        "tips": [
            _("The price is for one image"),
            _("Any languages"),
        ],
        "size_options": [
            {
                "width": 1000,
                "height": 1000,
                "cost": 6000,
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

async def inference(model, prompt, width, height):
    images = None
    if model == "dalle":
        image_url = await openai_utils.create_image(prompt)
        images = [image_url]
    elif model in sinkinai_utils.MODELS:
        images = await sinkinai_utils.inference(model=model, width=width, height=height, prompt=prompt)
    elif model in replicate_utils.MODELS:
        images = await replicate_utils.inference(model=model, width=width, height=height, prompt=prompt)
    elif model in getimg_utils.MODELS:
        images = await getimg_utils.inference(model=model, width=width, height=height, prompt=prompt)

    if images is None:
        raise Exception("invalid model")
    return images
