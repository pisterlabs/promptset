import time
import io
import gradio as gr
import cv2
import base64
import openai

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from PIL import Image

from prompts import VISION_SYSTEM_PROMPT, AUDIO_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, FINAL_EVALUATION_PROMPT


global_dict = {}

######
# SETTINGS
VIDEO_FRAME_LIMIT = 2000

######


def validate_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)

    try:
        # Make your OpenAI API request here
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Hello world"},
            ]
        )
    except openai.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        response = None
        error = e
        pass
    except openai.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        response = None
        error = e
        pass
    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        response = None
        error = e
        pass

    if response:
        return True
    else:
        raise gr.Error(f"OpenAI returned an API Error: {error}")


def _process_video(video_file):
    # Read and process the video file
    video = cv2.VideoCapture(video_file.name)

    if 'video_file' not in global_dict:
        global_dict.setdefault('video_file', video_file.name)
    else:
        global_dict['video_file'] = video_file.name

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    if len(base64Frames) > VIDEO_FRAME_LIMIT:
        raise gr.Warning(f"Video's play time is too long. (>1m)")
    print(len(base64Frames), "frames read.")

    if not base64Frames:
        raise gr.Error(f"Cannot open the video.")
    return base64Frames


def _make_video_batch(video_file, batch_size, total_batch_percent):

    frames = _process_video(video_file)

    TOTAL_FRAME_COUNT = len(frames)
    BATCH_SIZE = int(batch_size)
    TOTAL_BATCH_SIZE = int(TOTAL_FRAME_COUNT * total_batch_percent / 100)
    BATCH_STEP = int(TOTAL_FRAME_COUNT / TOTAL_BATCH_SIZE)
    
    base64FramesBatch = []

    for idx in range(0, TOTAL_FRAME_COUNT, BATCH_STEP * BATCH_SIZE):
        # print(f'## {idx}')
        temp = []
        for i in range(BATCH_SIZE):
            # print(f'# {idx + BATCH_STEP * i}')
            if (idx + BATCH_STEP * i) < TOTAL_FRAME_COUNT:
                temp.append(frames[idx + BATCH_STEP * i])
            else:
                continue
        base64FramesBatch.append(temp)
    
    for idx, batch in enumerate(base64FramesBatch):
        # assert len(batch) <= BATCH_SIZE
        print(f'##{idx} - batch_size: {len(batch)}')

    if 'batched_frames' not in global_dict:
        global_dict.setdefault('batched_frames', base64FramesBatch)
    else:
        global_dict['batched_frames'] = base64FramesBatch

    return base64FramesBatch


def show_batches(video_file, batch_size, total_batch_percent):
    
    batched_frames = _make_video_batch(video_file, batch_size, total_batch_percent)
    
    images = []
    for i, l in enumerate(batched_frames):
        print(f"#### Batch_{i+1}")
        for j, img in enumerate(l):
            print(f'## Image_{j+1}')
            image_bytes = base64.b64decode(img.encode("utf-8"))
            # Convert the bytes to a stream (file-like object)
            image_stream = io.BytesIO(image_bytes)
            # Open the image as a PIL image
            image = Image.open(image_stream)
            images.append((image, f"batch {i+1}"))
        print("-"*100)
    
    return images


def show_audio_transcript(video_file, api_key):
    previous_video_file = global_dict.get('video_file')

    if global_dict.get('transcript') and previous_video_file == video_file.name:
        return global_dict['transcript']
    else:
        audio_file = open(video_file.name, "rb")

        client = openai.OpenAI(api_key=api_key)
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )
        if 'transcript' not in global_dict:
            global_dict.setdefault('transcript', transcript)
        else:
            global_dict['transcript'] = transcript

        return transcript


def change_audio_rubric(choice):
    print(choice)
    if choice == "Video only":
        return gr.Textbox(
            visible=False
        )
    else:
        return gr.Textbox(
                    label="3. Audio Evaluation Rubric (if needed)",
                    info="Enter your evaluation rubric here...",
                    placeholder="<RUBRIC>\nHere's what the performer should *SAY* as follows:\n1. From standing, you need to shout 'Start' signal.\n2. Rock forward, you shouldn't make any noise while rolling.\n3. Standing still again, you need to shout 'Finish' signal.",
                    lines=7,
                    interactive=True,
                    visible=True)


def change_audio_eval(choice):
    print(choice)
    if choice == "Video only":
        return gr.Textbox(
            visible=False,
        )
    else:
        return gr.Textbox(
                    label="Audio Script Eval...",
                    lines=10,
                    interactive=False,
                    visible=True
                )


def call_gpt_vision(api_key, rubrics, progress=gr.Progress()) -> list:
    frames = global_dict.get('batched_frames')
    openai.api_key = api_key

    full_result_vision = []
    full_text_vision = ""
    idx = 0

    for batch in progress.tqdm(frames):
        VISION_PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": VISION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    PromptTemplate.from_template(USER_PROMPT_TEMPLATE).format(rubrics=rubrics),
                    *map(lambda x: {"image": x, "resize": 300}, batch),
                ],
            },
        ]
        
        params = {
        "model": "gpt-4-vision-preview",
        "messages": VISION_PROMPT_MESSAGES,
        "max_tokens": 1024,
        }

        try:
            result = openai.chat.completions.create(**params)
            print(result.choices[0].message.content)
            full_result_vision.append(result)
        except Exception as e:
            print(f"Error: {e}")
            full_text_vision += f'### BATCH_{idx+1}\n' + "-"*50 + "\n" + f"Error: {e}" +  "\n" + "-"*50 + "\n"
            idx += 1
            pass
        
        if 'full_result_vision' not in global_dict:
            global_dict.setdefault('full_result_vision', full_result_vision)
        else:
            global_dict['full_result_vision'] = full_result_vision
        
        print(f'### BATCH_{idx+1}')
        print('-'*100)
        full_text_vision += f'### BATCH_{idx+1}\n' + "-"*50 + "\n" + result.choices[0].message.content +  "\n" + "-"*50 + "\n"
        idx += 1
        time.sleep(2)

    return full_text_vision


def call_gpt_audio(api_key, rubrics) -> str:
    transcript = global_dict.get('transcript')
    openai.api_key = api_key

    full_text_audio = ""

    print(f"RUBRIC_AUDIO: {rubrics}")
    if not rubrics:
        return full_text_audio
    else:
        PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": AUDIO_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": PromptTemplate.from_template(USER_PROMPT_TEMPLATE).format(rubrics=rubrics) + "\n\n<TEXT>\n" + transcript
            },
        ]
        params = {
            "model": "gpt-4",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 1024,
        }

        try:
            result = openai.chat.completions.create(**params)
            full_text_audio = result.choices[0].message.content
            print(full_text_audio)
        except openai.OpenAIError as e:
            print(f"Failed to connect to OpenAI: {e}")
            pass

        if 'full_text_audio' not in global_dict:
            global_dict.setdefault('full_text_audio', full_text_audio)
        else:
            global_dict['full_text_audio'] = full_text_audio

        return full_text_audio


def get_full_result():
    full_result_vision = global_dict.get('full_result_vision')
    full_result_audio = global_dict.get('full_text_audio')
    
    result_text_video = ""
    result_text_audio = ""


    for idx, res in enumerate(full_result_vision):
        result_text_video += f'<Video Evaluation_{idx+1}>\n'
        result_text_video += res.choices[0].message.content
        result_text_video += "\n"
        result_text_video += "-"*5
        result_text_video += "\n"
    result_text_video += "*"*5 + "END of Video" + "*"*5 

    if full_result_audio:
        result_text_audio += '<Audio Evaluation>\n'
        result_text_audio += full_result_audio
        result_text_audio += "\n"
        result_text_audio += "-"*5
        result_text_audio += "\n"
        result_text_audio += "*"*5 + "END of Audio" + "*"*5 

        result_text = result_text_video + "\n\n" + result_text_audio
    else:
        result_text = result_text_video
    
    global_dict.setdefault('result_text', result_text)

    return result_text


def get_final_anser(api_key, result_text):
    chain = ChatOpenAI(
        api_key=api_key,
        model="gpt-4",
        max_tokens=1024,
        temperature=0,
    )
    prompt = PromptTemplate.from_template(FINAL_EVALUATION_PROMPT)
    runnable = prompt | chain | StrOutputParser()

    final_eval = runnable.invoke({"evals": result_text})
    return final_eval


# Define the Gradio app
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# GPT-4 Vision for Evaluation")
        gr.Markdown("## 1st STEP. Make Batched Snapshots & Audio Script")
        with gr.Row():
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(
                    label="Enter your OpenAI API Key",
                    info="Your API Key must be allowed to use GPT-4 Vision",
                    placeholder="sk-*********...",
                    lines=1
                )
                video_upload = gr.File(
                    label="Upload your video (under 1 minute video is the best..!)",
                    file_types=["video"],
                )
                batch_size = gr.Slider(
                    label="Number of images in one batch",
                    info="Choose between 2 and 5",
                    value=5,
                    minimum=2,
                    maximum=5,
                    step=1
                )
                total_batch_percent = gr.Slider(
                    label="Percentage(%) of batched image frames to total frames",
                    info="Choose between 1(%) and 5(%)",
                    value=3,
                    minimum=1,
                    maximum=5,
                    step=1
                )
                process_button = gr.Button("Process")         
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="Batched Snapshots of Video",
                    columns=[5],
                    rows=[2],
                    object_fit="contain",
                    height="auto",
                )
                transcript_box = gr.Textbox(
                    label="Audio Transcript",
                    info="(You can EDIT by yourself)",
                    lines=8,
                    interactive=True,
                )
        
        gr.Markdown("## 2nd STEP. Set Evaluation Rubric")
        with gr.Row():
            with gr.Column(scale=1):
                multimodal_radio = gr.Radio(
                    label="1. Multimodal Selection",
                    info="Choose evaluation channel",
                    value="Video + Audio",
                    choices=["Video + Audio", "Video only"]
                )
                rubric_video_input = gr.Textbox(
                    label="2. Video Evaluation Rubric",
                    info="Enter your evaluation rubric here...",
                    placeholder="Here's what the performer should *SHOW* as follows:\n1. From standing, bend your knees and straighten your arms in front of you.\n2. Place your hands on the floor, shoulder width apart with fingers pointing forward and your chin on your chest.\n3. Rock forward, straighten legs and transfer body weight onto shoulders.\n4. Rock forward on a rounded back placing both feet on the floor.\n5. Stand using arms for balance, without hands touching the floor.",
                    lines=7
                )
                rubric_audio_input = gr.Textbox(
                    label="3. Audio Evaluation Rubric (if needed)",
                    info="Enter your evaluation rubric here...",
                    placeholder="Here's what the performer should *SAY* as follows:\n1. From standing, you need to shout 'Start' signal.\n2. Rock forward, you shouldn't make any noise while rolling.\n3. Standing still again, you need to shout 'Finish' signal.",
                    interactive=True,
                    visible=True,
                    lines=7
                )            
                evaluate_button = gr.Button("Evaluate")
            with gr.Column(scale=1):
                video_output_box = gr.Textbox(
                    label="Video Batched Snapshots Eval...",
                    lines=8,
                    interactive=False
                )
                audio_output_box = gr.Textbox(
                    label="Audio Script Eval...",
                    lines=8,
                    interactive=False,
                    visible=True
                )

        gr.Markdown("## 3rd STEP. Summarize and Get Result")
        with gr.Row():
            with gr.Column(scale=1):
                output_box_fin = gr.Textbox(
                    label="FULL Response",
                    info="You can edit partial evaluation in here...",
                    lines=10,
                    interactive=True,
                )
                summarize_button = gr.Button("Summarize")

            with gr.Column(scale=1):
                output_box_fin_fin = gr.Textbox(
                    label="Final Evaluation",
                    lines=10,
                    interactive=True,
                    show_copy_button=True,
                )

        multimodal_radio.change(fn=change_audio_rubric, inputs=multimodal_radio, outputs=rubric_audio_input)
        multimodal_radio.change(fn=change_audio_eval, inputs=multimodal_radio, outputs=audio_output_box)

        process_button.click(fn=validate_api_key, inputs=api_key_input, outputs=None).success(fn=show_batches, inputs=[video_upload, batch_size, total_batch_percent], outputs=gallery).success(fn=show_audio_transcript, inputs=[video_upload, api_key_input], outputs=transcript_box)  
        if multimodal_radio.value == "Video + Audio":
            evaluate_button.click(fn=call_gpt_vision, inputs=[api_key_input, rubric_video_input], outputs=video_output_box).then(fn=call_gpt_audio, inputs=[api_key_input, rubric_audio_input], outputs=audio_output_box).then(get_full_result, None, output_box_fin)
        else:
            evaluate_button.click(fn=call_gpt_vision, inputs=[api_key_input, rubric_video_input], outputs=video_output_box).then(get_full_result, None, output_box_fin)
        summarize_button.click(fn=get_final_anser, inputs=[api_key_input, output_box_fin], outputs=output_box_fin_fin)

    demo.launch()

if __name__ == "__main__":
    main()
