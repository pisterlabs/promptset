from transformers import WhisperProcessor, WhisperForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import gradio as gr
import librosa
import openai
import numpy as np

# load model and processor for ASR
checkpoint_asr = "openai/whisper-tiny"
processor_asr = WhisperProcessor.from_pretrained(checkpoint_asr)
model_asr = WhisperForConditionalGeneration.from_pretrained(checkpoint_asr)
model_asr.config.forced_decoder_ids = None

# load model and processor for TTS
checkpoint_tts = "microsoft/speecht5_tts"
vocoder_tts = "microsoft/speecht5_hifigan"
processor_tts = SpeechT5Processor.from_pretrained(checkpoint_tts)
model_tts = SpeechT5ForTextToSpeech.from_pretrained(checkpoint_tts)
vocoder_tts = SpeechT5HifiGan.from_pretrained(vocoder_tts)

def process_audio(sampling_rate, waveform):
    # convert to float
    waveform = waveform / 32678.0

    # convert to mono
    if len(waveform.shape) > 1:
        waveform = librosa.to_mono(waveform.T)

    # resample to 16 kHz
    if sampling_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=16000)

    # make array
    waveform = np.array(waveform)
    return waveform

def transcript(audio):
    if audio is not None:
        sampling_rate, waveform = audio
    else:
        return "Start the recording!"

    waveform = process_audio(sampling_rate, waveform)
    
    input = processor_asr(audio=waveform, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model_asr.generate(input)
    transcription = processor_asr.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

def textToSpeech(text):
    if len(text.strip()) == 0:
        return (16000, np.zeros(0).astype(np.int16))

    speaker_embedding = np.load("speaker_embeddings/cmu_us_clb_arctic-wav-arctic_a0144.npy")
    speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0)

    inputs = processor_tts(text=text, return_tensors="pt")
    speech = model_tts.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder_tts)
    speech = (speech.numpy() * 32767).astype(np.int16)
    return (16000, speech)

def chat(openAI_key, audio):
    if openAI_key is not None:
        openai.api_key = openAI_key
    else:
        return "Provide a valid OpenAI API Key!"
    
    # Automatic Speech Recognition
    prompt = transcript(audio)

    # GPT gives an answer
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
        ]
    )
    generated_text = completion.choices[0].message["content"]

    # Text to Speech
    answer = textToSpeech(generated_text)
    return prompt, generated_text, answer

description = """
Your GPT-based vocal assistant. Speech recognition is performed with the <b>openai/whisper-tiny model</b>, while Text-to-Speech with <b>microsoft/speecht5_tts</b>.
<br>
<br>
References:<br>
<a href="https://huggingface.co/openai/whisper-tiny">OpenAI Whisper-tiny</a><br>
<a href="https://huggingface.co/microsoft/speecht5_tts">Microsoft SpeechT5_tts</a><br>
<a href="https://huggingface.co/blog/speecht5">Matthijs, Huggingface - Speech Synthesis, Recognition, and More With SpeechT5</a><br>
<a href="https://huggingface.co/docs/transformers/tasks/asr">Huggingface - ASR with Transformers</a>.<br>
<a href="https://platform.openai.com">OpenAI API Reference</a><br>
"""

gr.Interface(
    fn=chat,
    inputs=[
        gr.Text(label="Provide an OpenAI API Key"),
        gr.Audio(label="Record", source="microphone", type="numpy")
    ],
    outputs=[
        gr.Text(label="Transcription"),
        gr.Text(label="GPT Answer"),
        gr.Audio(label="Speech Answer", type="numpy")
    ],
    title="GIVA - GPT-based Interactive Vocal Agent",
    description=description
).launch()