import base64
from dotenv import load_dotenv

import openai
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app import crud


def vision_agent(
    db: Session,
    video_id: int,
    user_input: str,
    end_sec: int,
    context_len: int,
    reactor: str,
):
    # end_sec needs to be dynamically defined by identifying at which second the user started talking in the video

    # encode frames
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # get keys
    load_dotenv()

    start_sec = max(1, end_sec - context_len)
    frames_context = crud.frame.get(
        db=db, video_id=video_id, start_sec=start_sec, end_sec=end_sec
    )
    if not frames_context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video w/ id {video_id} has no frames from {start_sec} to {end_sec}",
        )

    for idx in range(len(frames_context)):
        file_path = frames_context[idx]
        if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            # Encode the image to base64 strings and replace
            frames_context[idx] = encode_image(file_path)

    res = crud.audiotext.get(
        db=db, video_id=video_id, start_sec=start_sec, end_sec=end_sec
    )
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video w/ id {video_id} has no audio transcript from {start_sec} to {end_sec}",
        )

    audio_context = " ".join(res)

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                f"You are a YouTube reactor like {reactor}. Your job is to be hype. React to the video content as you watch with your friend and respond to your friend if they say something. Given are the frames of a video from second {start_sec} to {end_sec}. During this time, the video also said: '{audio_context}'. Your friend said: '{user_input}'",
                *map(lambda x: {"image": x, "resize": 768}, frames_context),
            ],
        },
    ]

    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
    }

    result = openai.chat.completions.create(**params)
    text = result.choices[0].message.content

    response = openai.audio.speech.create(model="tts-1", voice="nova", input=text)

    return response, text
