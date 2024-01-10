import openai
from dataset import to_device, get_device, get_image
from model import get_model
import torch
import config


# Set up OpenAI API credentials
openai.api_key = "[Secret-Key]"


# function to call OpenAI API and get insights and recommendations
def get_recommendation(predicted_label):
    prompt = f"What are the best practices for optimizing yield in {predicted_label} land type?"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.8,
    )
    recommendation = response.choices[0].text.strip()
    #recommendation_points = recommendation.split(". ")
    #recommendation = "\n".join([f"{i}. {point}" for i, point in enumerate(recommendation_points, start=1)])


    return recommendation


def decode_target(target, text_labels=False):
    """Decode target labels into text (or not)"""
    if not text_labels:
        return target
    else:
        return config.IDX_CLASS_LABELS[target]


def predict_single(image_path):
    device = get_device()
    image = get_image(image_path, device)
    model = get_model(device)
    model.eval()
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    with torch.no_grad():
        preds = model(xb)
    _, prediction = torch.max(preds.cpu().detach(), dim=1)

    # Get insights and recommendations using OpenAI API
    predicted_label = decode_target(int(prediction), text_labels=True)
    recommendation = get_recommendation(predicted_label)
    recommendation = recommendation.replace('. ', '.\n')


    return predicted_label, recommendation
