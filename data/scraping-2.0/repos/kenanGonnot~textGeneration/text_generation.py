import openai
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel
import model as gpt


def generate_text_openai(text_length, starting_text, model_name="ada", temperature=0.8):
    """
    Générer du texte avec OpenAI
    :param text_length:
    :param starting_text:
    :param model_name: ada | babbage | curie | davinci
    :param temperature:
    :return: Le texte généré
    """
    # Générer le texte avec OpenAI
    response = openai.Completion.create(
        engine=model_name,
        prompt=starting_text,
        temperature=temperature,
        max_tokens=text_length,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n", " Human:", " AI:"]
    )
    return response['choices'][0]['text']


def generate_text_gpt2(text_length, starting_text, model_name="gpt2",
                       temperature=0.8):  # gpt2-large | gpt2-medium | gpt2
    """
    Générer du texte avec HuggingFace GPT-2
    :param text_length:
    :param starting_text:
    :param model_name:
    :param temperature:
    :return:
    """
    # Générer le texte avec HuggingFace GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id) #TFGPT2LMHeadModel for tensorflow

    encoded_input = tokenizer.encode(starting_text, return_tensors='pt')
    output = model.generate(encoded_input, max_length=text_length, do_sample=True, temperature=temperature)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_text_kenbot(text_length, starting_text, PATH):
    """
    Générer du texte avec notre propre modèle
    :param text_length:
    :param starting_text:
    :param temperature: pas utilisé maintenant
    :return:
    """
    model = gpt.GPTLanguageModel()
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()
    text_generated = model.get_text_generated(text_length, starting_text)
    return text_generated
