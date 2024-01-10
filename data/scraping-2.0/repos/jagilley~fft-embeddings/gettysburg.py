import openai
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from openai.embeddings_utils import get_embeddings
import librosa
import pickle
import os
import soundfile as sf

with open('/users/jasper/oai.txt', 'r') as f:
    openai.api_key = f.read()

def split_into_chunks(text, num_words=5):
    words = text.split()
    chunks = []
    for i in range(0, len(words)):
        chunk = ' '.join(words[i:i+num_words])
        # if not end of text
        if i+num_words <= len(words):
            chunks.append(chunk)
    return chunks

text_to_load = 'texts/gettysburg.txt'
with open(text_to_load, 'r') as f:
    gettysburg = f.read()

chunks = split_into_chunks(gettysburg, num_words=10)

embeddings_path = f'embeddings/{text_to_load.split("/")[-1].replace(".txt", ".pkl")}'
# Get embeddings for each chunk
if os.path.exists(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
else:
    embeddings = get_embeddings(chunks)
    # pickle embeddings
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)

# embeddings to numpy
embeddings = np.array(embeddings)
print(embeddings.shape)

# get spectrogram for each channel
for i in range(0, 1):
    # continue
    channel_0 = embeddings[:, i]

    # Plot channel 0
    # plt.plot(channel_0)
    # plt.show()

    # compute spectrogram for channel 0
    spec = np.abs(librosa.stft(channel_0, n_fft=32, win_length=4))
    print(spec.shape)

    # Plot spectrogram
    plt.imshow(spec)
    # label axes
    plt.xlabel('Time, in frames')
    plt.ylabel('Frequency')

    plt.show()

    # convert to audio
    # fill in material such that we have a 5 second audio clip
    # do this by interpolating between each frame

    # audio = librosa.istft(spec, win_length=4)
    # # normalize audio
    # audio = audio / np.max(np.abs(audio))
    # # save audio
    # sf.write(f'audio_{i}.wav', audio, 22050)

# get spectrogram for all channels
spec = np.abs(librosa.stft(embeddings, n_fft=32, win_length=4))

import code
code.interact(local=locals())