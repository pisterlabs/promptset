import uuid
from midiutil import MIDIFile
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import FileResponse
import openai

from api.utils.sonify import (process_frame,
                              sonify_image,
                              write_midi_file,
                              convert_midi_to_mp3,
                              compute_luminance,
                              apply_gamma_correction,
                              sonify_pixel, TEMPO
                              ,check_tuning_file)
from api.utils.decompose import decompose_img


app = FastAPI()

openai.api_key = "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"

utils_dir = "./api/utils/"

@app.post("/sonify")
async def sonfiy(media: UploadFile = File(...), melody: UploadFile = File(None)):
    video = True
    if media.content_type in ["image/jpeg", "image/png", "image/jpg", "image/gif", "image/bmp", "image/webp"]:
        video = False
    elif media.content_type in ["video/mp4"]:
        video = True
    else:        
        raise HTTPException(status.HTTP_409_CONFLICT, "image must be of jpeg, png, jpg, gif, bmp or webp type\n or video of mp4")
    
    if melody:
        if melody.content_type not in ["audio/mid"]:
            raise HTTPException(status.HTTP_409_CONFLICT, "melody must be of midi type")
        with open(utils_dir + melody.filename, "wb") as f:
            melody_contents = await melody.read()
            f.write(melody_contents)
    
    general_name = f"{uuid.uuid4()}"
    media.filename = general_name + "." + media.filename.split(".")[1]
    
    with open(utils_dir + media.filename, "wb") as f:
        media_contents = await media.read()
        f.write(media_contents)
    
    down_scaled_image = process_frame(utils_dir+media.filename)
    MIN_VOLUME, MAX_VOLUME, note_midis = check_tuning_file(utils_dir + melody.filename if melody else "") 
    print(MIN_VOLUME, MAX_VOLUME, note_midis)       
    midi_file = sonify_image(down_scaled_image, MIN_VOLUME, MAX_VOLUME, note_midis)
    write_midi_file(midi_file, utils_dir + general_name)
    convert_midi_to_mp3(utils_dir+f"{general_name}.mid", utils_dir+ "sound-font.sf2", utils_dir+ f"{general_name}.mp3")
    
    return FileResponse(utils_dir+ f"{general_name}.mp3")

        
@app.post("/color_tone")
async def get_color_tone(hex: str):
    hex = hex.lstrip("#")
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    
    # if melody:
    #     if melody.content_type not in ["audio/mid"]:
    #         raise HTTPException(status.HTTP_409_CONFLICT, "melody must be of midi type")
    #     with open(utils_dir + melody.filename, "wb") as f:
    #         melody_contents = await melody.read()
    #         f.write(melody_contents)
    
    # MIN_VOLUME, MAX_VOLUME, note_midis = check_tuning_file(utils_dir + melody.filename if melody else "") 
    MIN_VOLUME, MAX_VOLUME, note_midis = check_tuning_file("") 
    
    
    luminance = compute_luminance(rgb)
    pitch, duration, volume = sonify_pixel(rgb, luminance, MAX_VOLUME, MAX_VOLUME, note_midis)
    midi_filename = str(rgb[0]) + "-" + str(rgb[1]) + "-" + str(rgb[2])
    midi_file = MIDIFile(1)
    midi_file.addTempo(track=0, time=0, tempo=TEMPO)  # add midi notes
    midi_file.addNote(
                track=0,
                channel=0,
                time=0,
                pitch=pitch,
                volume=volume,
                duration=duration,
            )
    write_midi_file(midi_file, utils_dir+ midi_filename)
    convert_midi_to_mp3(utils_dir+ f"{midi_filename}.mid", utils_dir+ "sound-font.sf2", utils_dir+ f"{midi_filename}.mp3")
    
    return FileResponse(utils_dir+ f"{midi_filename}.mp3")

@app.post("/decompose")
async def decompose(image: UploadFile = File(...)):
    with open(utils_dir + image.filename, "wb") as f:
        media_contents = await image.read()
        f.write(media_contents)
        
    return decompose_img()

def chat_with_chatgpt(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"content": prompt, "role": "user"}],
        temperature=0,
    )

    message = response['choices'][0]['message']['content']
    return message

@app.get("/chatgpt")
def get_chatgpt(propmt: str):
    return chat_with_chatgpt(propmt)
