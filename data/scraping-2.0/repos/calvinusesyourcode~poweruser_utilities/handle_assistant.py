from openai import OpenAI
import json
import time

def show_json(obj):
    print(json.loads(obj.model_dump_json()))

def wait_on_run(client, run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


# client = OpenAI()


# thread = client.beta.threads.create()
# show_json(thread)

# message = client.beta.threads.messages.create(
#     thread_id=thread.id,
#     role="user",
#     content="""Turn this file into an mp3 of moderate audio quality "C:\\Users\\calvi\\3D Objects\\poweruser_utilities\\downloads\\audio_____creative_exercise__mario_paint_music_extended.mp3" """,
# )
# show_json(message)

# run = client.beta.threads.runs.create(
#     thread_id=thread.id,
#     assistant_id=assistant_id,
# )
# show_json(run)

# run = wait_on_run(run, thread)
# show_json(run)

# messages = client.beta.threads.messages.list(thread_id=thread.id)
# show_json(messages)

# for message in messages.data:
#     if message.content and len(message.content) > 0:
#         for content in message.content:
#             if content.text and "ffmpeg" in content.text.value:
#                 ffmpegCommand = content.text.value
#                 break
#     if ffmpegCommand:
#         break

# print(ffmpegCommand)

def perform_assistant_run(assistant_id, prompt):   
    client = OpenAI()
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    run = wait_on_run(client, run, thread)
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    result = None
    for message in messages.data:
        if message.content and len(message.content) > 0:
            for content in message.content:
                if content.text and content.text.value:
                    result = content.text.value
                    break
        if result:
            break
    return result

def assistant_test():
    prompt = """Turn this file into an mp3 of moderate audio quality "C:\\Users\\calvi\\3D Objects\\poweruser_utilities\\downloads\\audio_____creative_exercise__mario_paint_music_extended.mp3" """
    assistant_id = "asst_7Fn6Z2FWiINxpC5ay3q5QiJN"
    result = perform_assistant_run(assistant_id, prompt)
    print(result)

def ffmpeg_assist():
    import os, pyperclip, subprocess
    assistant_id = "asst_7Fn6Z2FWiINxpC5ay3q5QiJN"
    path = pyperclip.paste()
    if "." not in path:
        raise Exception("Doesn't seem to be a file extension...")
    if os.name == 'nt':
        path = path.replace('\\', '\\\\')
    folder = os.path.dirname(path).replace('"','').replace("'",'')
    basename = os.path.basename(path).replace('"','').replace("'",'')

    prompt = input("prompt: ")
    prompt += f' "{basename}"'
    result = perform_assistant_run(assistant_id, prompt)
    print(result)
    os.chdir(folder)
    subprocess.run(result, shell=True)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

# assistant_test()
# ffmpeg_assist()