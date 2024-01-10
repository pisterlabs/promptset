from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import json
import openai
from models.model import Config, Model
from getInput import get_model_input


def convert(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError


# select mode path here
# see more at https://huggingface.co/kornosk
def predict(text, target, language='en'):
    config = Config(language)
    # load model
    model = Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location=torch.device('cpu')))
    model.eval()
    # get inputs in a model-acceptable form
    inputs = get_model_input(text, target, config)
    # inputs
    outputs = model(inputs)
    predicted_probability = torch.softmax(outputs, dim=1)
    predicted_probability = predicted_probability[0].tolist()
    print(predicted_probability)
    id2label = {
        0: "FAVOR",
        1: "AGAINST",
        2: "NEITHER"
    }
    max_prob = max(predicted_probability)
    index = predicted_probability.index(max_prob)
    predicted_label = id2label[index]
    probs = {
        'Favor': predicted_probability[0],
        'Against': predicted_probability[1],
        'Neither': predicted_probability[2]
    }
    probs_json_obj = json.dumps(probs, default=convert)
    prediction_data = {"label": predicted_label,
                       "max_prob": max_prob, "probs": probs_json_obj}
    json_obj = json.dumps(prediction_data, default=convert)
    return json_obj, (predicted_label, probs)


if __name__ == "__main__":
    language = 'en'
    # language = 'ch'
    if language == 'en':
        target = 'Hillary Clinton'
        sentence = "@AnthonyCumia: Hillary is an idiot!!! If she IS elected, I have my transition team together to smoothly move me from a racist to a sexist"
    elif language == 'ch':
        target = 'iPhoneSE'
        sentence = '很好的手机，性价比高，比国产好的没话说，国产一年一环 苹果用 5 年 8 年没事不卡不迟钝，黑粉们哪个牌子派来的去哪里带着去，有意思吗？不喜欢就不要买，到处乱喷 这就是素质'
    result = predict(target, sentence,language)
    print(result)
