import openai
from docx import Document
import os
import soundfile as sf
import numpy as np
import streamlit as st
import time
from pydub import AudioSegment
from moviepy.editor import *
import math
from openai import OpenAI
import csv
import ast



openai.api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI()

# Streamlit Configuration
st.set_page_config(
    page_title="LV FNOL Test",
    page_icon="https://th.bing.com/th/id/OIP.IPordwq_aME0Zl6Fa2GYagAAAA?rs=1&pid=ImgDetMain"
)

def get_state_variable(var_name, default_value):
    if 'st_state' not in st.session_state:
        st.session_state['st_state'] = {}
    if var_name not in st.session_state['st_state']:
        st.session_state['st_state'][var_name] = default_value
    return st.session_state['st_state'][var_name]

# Initialize session state for authentication
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False


def display_page():

    # Function to split audio
    def split_audio_sf(audio_path, target_chunk_size, max_file_size=25 * 1024 * 1024):
        # Read the audio data
        data, samplerate = sf.read(audio_path)
        total_samples = len(data)

        while True:
            chunk_samples = int(target_chunk_size * samplerate)

            # Check if the chunk size produces files larger than max_file_size
            num_chunks = math.ceil(total_samples / chunk_samples)
            print(f"Splitting audio into {num_chunks} chunks of {chunk_samples} samples each")
            estimated_size_per_chunk = os.path.getsize(audio_path) / num_chunks
            print(f"Estimated size per chunk: {estimated_size_per_chunk} bytes")

            if estimated_size_per_chunk > max_file_size:
                # If chunks are too large, reduce the target chunk size and recalculate
                target_chunk_size -= 10  # Reduce by 10 seconds and retry
                if target_chunk_size <= 0:
                    raise ValueError("Cannot split audio into chunks smaller than the max file size.")
                continue  # If still too large, loop back and try with reduced chunk size

            # If the estimated size per chunk is within the limit, proceed to create chunks
            chunks = [(data[start:start + chunk_samples], samplerate) for start in range(0, total_samples, chunk_samples)]
            break  # If chunk sizes are okay, break out of the loop and return chunks

        return chunks

    def transcribe_chunks_sf(chunks):
        transcriptions = []
        
        for i, (data, samplerate) in enumerate(chunks):
            temp_path = f'temp_chunk_{i}.wav'
            sf.write(temp_path, data, samplerate)

            # Check file size
            file_size = os.path.getsize(temp_path)
            if file_size > 26214400:  # 25 MB in bytes
                print(f"Skipping chunk {i} due to large size: {file_size} bytes")
                os.remove(temp_path)
                continue

            transcription = transcribe_audio(temp_path)
            transcriptions.append(transcription)
            os.remove(temp_path)
            
        return ' '.join(transcriptions)


    def transcribe_audio(audio_file_path):
        with open(audio_file_path, 'rb') as audio_file:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                language="en"  # Set the language to English
            )
        return transcription.text  # Here, we use .text instead of ['text']



    def meeting_minutes(transcription):
        word_count = len(transcription.split())
        # Calculate sleep duration: 4 seconds for every 1000 words
        sleep_duration = (word_count / 1000) * 1

        # Ensure minimum sleep time (e.g., 2 seconds) if transcription is very short
        sleep_duration = max(sleep_duration, 2)

        # Sleep dynamically based on the number of words in the transcription
        time.sleep(sleep_duration)

        abstract_summary = parse_text(transcription)
        time.sleep(sleep_duration)  # Repeat sleep after each processing step

        return {
            'abstract_summary': abstract_summary,
        }
        
    def parse_text(transcription):
        prompt = (
            "Analyze the following transcription and output the extracted information as a structured dictionary with keys and values. "
            "The keys are 'Claim Ref Number', 'Claimant', 'Images Requested', 'Mobile Number', 'Vehicle Registration', 'Make', 'Model', 'Incident Date', 'Incident Details', 'What Happened', 'Vehicle Status', 'Damage Description', 'Replacement Vehicle Required', and 'Excess Payable'. "
            "Each key should correspond to a value extracted from the transcription. If multiple areas are damaged format the value as a text string listing them. Do not use any single quotation marks within the strings, as this will cause a syntax error.\n\n"
            f"Please format the extracted information in a clear and structured manner as shown in the examples below.\n"
            "Example 1:\n"
            "{\n"
            "    'Claim Ref Number': '130-01-098978',\n"
            "    'Claimant': 'Mrs. Jasmine Franklin',\n"
            "    'Images Requested': 'No',\n"
            "    'Mobile Number': '08902332303',\n"
            "    'Vehicle Registration': 'NY68PWM',\n"
            "    'Make': 'Kia',\n"
            "    'Model': 'SPORTAGE 2 GDI',\n"
            "    'Incident Date': '3 December 2023 20:00'\n"
            "    'Incident Details': 'Hit Animal',\n"
            "    'What Happened': 'PH was driving along when a dog has ran and and PH has hit into.Police passed by and blocked the road. The dog had ran out the owners home and onto the road, the dog was taken to the vets and sadly passsed away.',\n"
            "    'Vehicle Status': 'Not Roadworthy, Secure',\n"
            "    'Damage Description': 'Front Offside Damage - Bumper, Wheel Arch.',\n"
            "    'Replacement Vehicle Required': 'Yes',\n"
            "    'Excess Payable': '600.00'\n"
            "}\n\n"
            "Example 2:\n"
            "{\n"
            "    'Claim Ref Number': '930-01-0409274',\n"
            "    'Claimant': 'Miss Robin Stevens',\n"
            "    'Images Requested': 'No',\n"
            "    'Mobile Number': '0975603618',\n"
            "    'Vehicle Registration': 'SE15MYI',\n"
            "    'Make': 'Toyota',\n"
            "    'Model': 'AYGO X-PLAY VVT-I',\n"
            "    'Incident Date': '27 November 2023 09:15'\n"
            "    'Incident Details': 'Third Party Hit Insured',\n"
            "    'What Happened': 'PH went out this morning and realised her car had been hit as her car has been damaged and vehicle doesnt go into reverse. TP is unknown and PH not at the scene when this incident occurred so PH not sure how this has happened.',\n"
            "    'Vehicle Status': 'Mobile, Roadworthy, Secure',\n"
            "    'Damage Description': 'Front Nearside Damage - Headlights, Wing, Wheel/Tyre/Alloy, Wheel Arch. Other Damage - Mechanical.',\n"
            "    'Replacement Vehicle Required': 'Yes',\n"
            "    'Excess Payable': '270.00'\n"
            "}\n\n"
            "Example 3:\n"
            "{\n"
            "    'Claim Ref Number': '420-84-996873',\n"
            "    'Claimant': 'Mr. Jeremy Rodgers',\n"
            "    'Images Requested': 'Yes',\n"
            "    'Mobile Number': '0899552133',\n"
            "    'Vehicle Registration': 'HU14TJM',\n"
            "    'Make': 'VOLKSWAGEN',\n"
            "    'Model': 'GOLF GTI PERFORMANCE TSI 230 AUTO',\n"
            "    'Incident Date': '12 November 2023 02:30'\n"
            "    'Incident Details': 'Third Party Hit Insured',\n"
            "    'What Happened': 'PHV was parked outside his house when TP has come round a corner and gone into PHV.PH was in bed at the time. Police and Ambulance was on scene.',\n"
            "    'Vehicle Status': 'Immobile, Not Roadworthy, Secure',\n"
            "    'Damage Description': 'Rear Offside Damage - Bumper, Wing, Wheel/Tyre/Alloy.',\n"
            "    'Replacement Vehicle Required': 'No',\n"
            "    'Excess Payable': '0.00'\n"
            "}\n\n"
            "Now, based on the transcription provided, format the extracted information in a similar structured dictionary:\n\n"
            f"Transcription:\n\"{transcription}\""
            "Provide ONLY the extracted information as a structured dictionary with keys and values. Do not include any other text or information."
        )
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2000,
        )
        # Access the content directly from the response object's attributes
        summary_content = response.choices[0].message.content
        return summary_content
    
    def save_as_csv(abstract_summary, filename):
    # Open the file in write mode
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            # Create a CSV writer object
            writer = csv.writer(file)

            # Write the headers (dictionary keys)
            headers = abstract_summary.keys()
            writer.writerow(headers)

            # Write the row of data (dictionary values)
            row = abstract_summary.values()
            writer.writerow(row)

    # Parameters
    chunk_length = 2 * 60  # 3 minutes in seconds

    def main():
        # Initialize Streamlit interface
        #st.title('Call Transcription and Generation')




        # Create columns for the image, title, and page selector
        col_img, col_title = st.columns([1, 3])

        # Upload the image to the left-most column
        with col_img:
            st.image("https://s3-eu-west-1.amazonaws.com/tpd/logos/5a95521f54e2c70001f926b8/0x0.png")


        # Set the title in the middle column based on page selection
        with col_title:
            st.header("FNOL Generator")       




        # File uploader allows user to add their own audio
        uploaded_audio = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

        if uploaded_audio is not None:
            # Button to trigger transcription and processing
            if st.button('Transcribe and Analyze'):
                # Process the audio
                with st.spinner('Processing audio...'):
                    audio_path = save_uploaded_file(uploaded_audio)

                    # Split audio
                    with st.spinner('Splitting audio into chunks...'):
                        audio_chunks = split_audio_sf(audio_path, chunk_length)

                    # Transcribe each chunk and concatenate
                    with st.spinner('Transcribing audio... this might take a while'):
                        full_transcription = transcribe_chunks_sf(audio_chunks)

                    # Delete temp file
                    if os.path.exists(audio_path):
                        os.remove(audio_path)

                    # Display transcription
                    st.subheader("Transcription")
                    st.text_area("Full Transcription", full_transcription, height=300)

                    # Generate meeting minutes
                    with st.spinner('Generating formatted data... this might take a while'):
                        minutes = meeting_minutes(full_transcription)

                        # Display meeting minutes details
                        st.subheader("Call Details")
                        for key, value in minutes.items():
                            st.markdown(f"#### {key.replace('_', ' ').title()}")
                            st.write(value)
                            abstract_summary=value
                            abstract_summary_dict = ast.literal_eval(abstract_summary)
                            st.write(abstract_summary_dict)

                        # Define the CSV file path
                        csv_file_path = 'call_details.csv'

                        # Save the data as a CSV file
                        save_as_csv(abstract_summary_dict, csv_file_path)

                        # Provide download link for the CSV file
                        with open(csv_file_path, "rb") as file:
                            btn = st.download_button(
                                label="Download formatted data as CSV",
                                data=file,
                                file_name=csv_file_path,
                                mime="text/csv"
                            )

    def convert_to_mp3(audio_path, target_path, bitrate="128k"):
        try:
            audio_clip = AudioFileClip(audio_path)
            audio_clip.write_audiofile(target_path, bitrate=bitrate)
            audio_clip.close()  # It's good practice to close the clip when done
            return True
        except Exception as e:
            print(f"Failed to convert {audio_path} to MP3: {e}")
            return False

    def save_uploaded_file(uploaded_file):
        try:
            file_extension = os.path.splitext(uploaded_file.name.lower())[1]
            base_name = os.path.splitext(uploaded_file.name)[0]
            save_folder = 'temp_files'

            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            temp_path = os.path.join(save_folder, uploaded_file.name)
            target_path = os.path.join(save_folder, f"{base_name}.mp3")

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                print(f"File saved temporarily at {temp_path}")

            # If the uploaded file is .m4a, convert it to .mp3
            if file_extension == '.m4a':
                print(f"Converting {temp_path} to {target_path}")
                # Verify tempfile exists before attempting conversion
                if os.path.exists(temp_path):
                    convert_to_mp3(temp_path, target_path)
                    # Verify conversion was successful
                    if os.path.exists(target_path):
                        print(f"Conversion successful, file saved at {target_path}")
                        os.remove(temp_path)  # Remove the .m4a file after conversion
                        return target_path
                    else:
                        print(f"Conversion failed, file at {target_path} not found")
                        return None
                else:
                    print(f"Temp file at {temp_path} does not exist, cannot convert")
                    return None
            elif file_extension in ['.mp3', '.wav']:
                return temp_path

        except Exception as e:
            st.error(f"Error handling file: {e}")
            return None

    # start the Streamlit app
    if __name__ == "__main__":
        main()







# Logic for password checking
def check_password():
    if not st.session_state.is_authenticated:
        password = st.text_input("Enter Password:", type="password")


            
        
        if password == st.secrets["db_password"]:
            st.session_state.is_authenticated = True
            st.rerun()
        elif password:
            st.write("Please enter the correct password to proceed.")
            
        blank, col_img, col_title = st.columns([2, 1, 3])

        # Upload the image to the left-most column
        with col_img:
            st.image("https://s3-eu-west-1.amazonaws.com/tpd/logos/5a95521f54e2c70001f926b8/0x0.png")


        # Determine the page selection using the selectbox in the right column
        with col_title:
            #st.title("Created By Halo")
            st.write("")
            st.markdown('<div style="text-align: left; font-size: 40px; font-weight: normal;">Created By Halo*</div>', unsafe_allow_html=True)
            
        blank2, col_img2, col_title2 = st.columns([2, 1, 3])

        # Upload the image to the left-most column
        with col_img2:
            st.image("https://d2t2wfirfyzjhs.cloudfront.net/images/ex-desc/lv-logo.png")


        # Determine the page selection using the selectbox in the right column
        with col_title2:
            
            #st.title("Powered By IRS")
            st.markdown('<div style="text-align: left; font-size: 30px; font-weight: normal;">Powered By LV</div>', unsafe_allow_html=True)
        # Fill up space to push the text to the bottom
        for _ in range(20):  # Adjust the range as needed
            st.write("")

        # Write your text at the bottom left corner
        st.markdown('<div style="text-align: right; font-size: 10px; font-weight: normal;">* Trenton Dambrowitz, Special Projects Manager, is "Halo" in this case.</div>', unsafe_allow_html=True)



    else:
        print("Access granted, welcome to the app.")
        display_page()


check_password()
