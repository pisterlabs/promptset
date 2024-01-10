from custom_riffusion import riffusion
from custom_riffusion.spectrogram.spectrogram_params import SpectrogramParams
from util.classifier import Classifier

import torch
import openai
import json
import numpy as np

openai.api_key = json.load(open("music_gen/py/conf.json"))['openai-api-key']

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def generate_music(positiv_prompt: str, negative_prompt: str, output_path: str,
                   format :str, num_steps: int = 20, seed: int = 42) -> np.array:
    # Generate music using Riffusion (https://github.com/riffusion/riffusion)
    img = riffusion.run_txt2img(
        prompt=positiv_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance=7.0,
        seed=seed,
        width=512,
        height=512,
        checkpoint="riffusion/riffusion-model-v1",
        device=device,
        scheduler="DPMSolverMultistepScheduler"
    )
    audio = riffusion.audio_segment_from_spectrogram_image(
        image=img,
        params=SpectrogramParams(
            stereo=False,
            sample_rate=44100,
            step_size_ms=10,
            window_duration_ms=100,
            padded_duration_ms=400,
            num_frequencies=512,
            min_frequency=0,
            max_frequency=10000,
            mel_scale_norm=None,
            mel_scale_type="htk",
            max_mel_iters=200,
            num_griffin_lim_iters=32,
            power_for_image=0.25,
        ),
        device=device
    )
    # Save music to file
    audio.export(output_path, format=format)
    return np.array(img)

def classify_music(path: str) -> Classifier:
    # Classify music using YAMNet
    classifier = Classifier()
    classifier.classify_single(path)
    return classifier

'''
# If the music is not classified as blues, try again with a different seed
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a music expert and are going to certify a music."},
        {"role": "user", "content": "Do the styles ['Keyboard (musical)', 'Music', 'Sampler', 'Cacophony', 'Musical instrument'] match the prompt 'a dark blues with drum'?"},
        {"role": "assistant", "content": "No"},
        {"role": "user", "content": "Do the styles ['Zither', 'Musical instrument', 'Mandolin', 'Pizzicato', 'Plucked string instrument'] match the prompt 'Electonic music with an hard keyboard and a hard drum'?"},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": f"Do the styles {classifier.get_datas()[0]['sub_class']} match the prompt {prompt} ?"},
    ]
)
assistant_reply = response['choices'][0]['message']['content'] # type: str
print(assistant_reply)
if assistant_reply.find("Yes") != -1 or assistant_reply.find("yes") != -1:
    print(colorama.Fore.GREEN + "The music is certified." + colorama.Style.RESET_ALL)
    break
i += 1
if i > iterations:
    break
# Print in red
print(colorama.Fore.RED + f"Try again with a different seed. (Interation {i})" + colorama.Style.RESET_ALL)
'''

def create_lirycs(prompt:str, neg_prompt:str, spectrogram: np.ndarray) -> str:
    # Create lirycs on the spectrogram using GPT
    message = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a music expert and a singer."},
            {"role": "user", "content": f"Give me lirycs on a music with this prompt '{prompt}', this negativ prompt '{neg_prompt}' and this spectrogam:\n\n{spectrogram}"},
        ],
    )
    return message['choices'][0]['message']['content']

if __name__ == "__main__":
    prompt = "Hard rock with a dark blues and a hard drum"
    neg_prompt = "classic music with a soft piano and a soft drum"
    img = generate_music(prompt, neg_prompt, r"music_gen\output\music.wav", "wav", 1) # type: np.ndarray
    classifier = classify_music("music_gen\output\music.wav")
    print(classifier)
    lirycs = create_lirycs(prompt, neg_prompt, img)
    print(lirycs)