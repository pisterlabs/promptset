import streamlit as st
import torch
import io
import openai
import os
import numpy as np
import soundfile as sf
from audiocraft.models import MusicGen, MultiBandDiffusion

# Fetch the OpenAI API Key securely
open_AI_key = os.environ.get('OPENAI_API_KEY')
if not open_AI_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Model Initialization
USE_DIFFUSION_DECODER = False
model = MusicGen.get_pretrained('facebook/musicgen-small')
if USE_DIFFUSION_DECODER:
    mbd = MultiBandDiffusion.get_mbd_musicgen()

model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=5
)

def generate_audio(result_text):
    output = model.generate(
        descriptions=[result_text],
        progress=True, return_tokens=True
    )
    audio_data = output[0]
    out_diffusion = None
    if USE_DIFFUSION_DECODER:
        out_diffusion = mbd.tokens_to_wav(output[1])
    return audio_data, out_diffusion

def numpy_to_bytes(audio_data, sample_rate):
    buffer = io.BytesIO()
    audio_data_2d = np.squeeze(audio_data)  # Remove singleton dimensions
    sf.write(buffer, audio_data_2d.T, sample_rate, format='WAV')  # Transpose if necessary
    buffer.seek(0)
    return buffer.getvalue()

# Streamlit App
def main():
    st.title("Fitness Activity Audio Generator")

    # User Inputs
    st.subheader("Activity Information")
    activity_date = st.text_input("Activity Date", value='2023-10-29')
    start_time = st.text_input("Start Time", value='12:00:00')
    end_time = st.text_input("End Time", value='12:10:00')
    activity_type = st.text_input("Type", value='Walking')
    duration = st.text_input("Duration (seconds)", value='600')
    distance = st.text_input("Distance (meters)", value='300')
    calories_burned = st.text_input("Calories Burned", value='20')
    avg_heart_rate = st.text_input("Average Heart Rate", value='80')
    peak_heart_rate = st.text_input("Peak Heart Rate", value='90')
    steps = st.text_input("Steps", value='400')
    notes = st.text_area("Notes", value='I feel tired and unmotivated.')

    # Action Button
    if st.button("Generate Audio"):
        user_input = f"""
        Activity Date: {activity_date}
        Start Time: {start_time}
        End Time: {end_time}
        Type: {activity_type}
        Duration: {duration}
        Distance: {distance}
        Calories Burned: {calories_burned}
        Average Heart Rate: {avg_heart_rate}
        Peak Heart Rate: {peak_heart_rate}
        Steps: {steps}
        Notes: {notes}
        """ 

        # OpenAI API Call
        openai.api_key = open_AI_key
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=(
                    "Generate a text based on what this person's activity shows. "
                    "If it has a negative implication, suggest a positive statement "
                    "to help and encourage them to recover from the negative situation.\n"
                    f"Response: {user_input}"
                ),
                max_tokens=50
            )
            result_text = response.choices[0].text.strip()
        except Exception as e:
            st.error("Error in OpenAI API call: " + str(e))
            return

        # Generating and displaying audio
        try:
            audio_data, out_diffusion = generate_audio(result_text)
            st.audio(numpy_to_bytes(audio_data, 32000), format='audio/wav')
            if out_diffusion:
                st.audio(numpy_to_bytes(out_diffusion, 32000), format='audio/wav')
        except Exception as e:
            st.error("Error in audio generation: " + str(e))

# Running the Streamlit app
if __name__ == "__main__":
    main()