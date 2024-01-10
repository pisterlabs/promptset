import base64
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
import requests
import streamlit as st
from st_audiorec import st_audiorec


from utils import get_function_body

# Load environment variables
load_dotenv()

@st.cache_resource
def get_client():
    return OpenAI()


client = get_client()


st.title("🧑‍🏫 Introduction to OpenAI (Part 2 - Artificial Boogaloo)")

"""
## Moar Chat Completion API

### 📞 Add Function Calling Tools

You can actually add tools that give additional powers to your LLM applications.

Think of this like adding new abilities.

It can literally do anything you can write code to do. Think APIs. 

💡 I'm going to try and cause a lightbulb moment.
"""

st.info("If you own [LIFX lights](https://www.lifx.com/) you can register your `LIFX_TOKEN` in your `.env` file.")

def make_light_request(payload):
    token = os.environ.get("LIFX_TOKEN", None)
    # Only try if the user has set the LIFX token
    if token is None:
        return None
    headers = {
        "Authorization": "Bearer %s" % token,
    }
    return requests.put(
        "https://api.lifx.com/v1/lights/all/state", data=payload, headers=headers
    )


if "current_color" not in st.session_state:
    st.session_state.current_color = "#FFFFFF"


def change_light_color(color, device_name="all"):
    print(f"Attempting to change {device_name} to {color}")
    st.session_state.current_color = color
    response = make_light_request({"color": color, "power": "on"})
    if response:
        print(response)
    return {"status": "success", "color": color}


def turn_lights_off():
    print("Attempting to turn lights off")
    st.session_state.current_color = "#000000"
    response = make_light_request({"power": "off"})
    if response:
        print(response)
    return {"status": "success", "power": "off"}


with st.form("lightchanger"):
    st.write(
        f"""`{st.session_state.current_color}`
<style>
    .bulb {{
        width: 50px;
        height: 50px;
        border: 1px solid black;
        border-radius: 50%;
        background-color: {st.session_state.current_color};
    }}
</style>
<div class="bulb" />
""",
        unsafe_allow_html=True,
    )
    user_input = st.text_input("Lighting Instructions")
    submitted = st.form_submit_button("Send instructions")
    messages = [
        {
            "role": "user",
            "content": user_input,
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "change_light_color",
                "description": "Changes the color of an IOT lightbulb",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "color": {
                            "type": "string",
                            "description": "A hex color value, prefixed with a #",
                        },
                        "device_name": {
                            "type": "string",
                            "description": "A named device like 'Bedroom Lamp'",
                        },
                    },
                    "required": ["color"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "turn_lights_off",
                "description": "Turns the lights off",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    if "light_message" in st.session_state:
        st.info(st.session_state.light_message)
    if "light_tool_calls" in st.session_state:
        with st.expander("Tool Calls (educational purposes)"):
            # This is just for display purposes
            calls_as_json_objects = [
                json.loads(c.model_dump_json())
                for c in st.session_state.light_tool_calls
            ]
            st.json(calls_as_json_objects)
    with st.expander("Defined Tools (educational purposes)"):
        st.json(tools)
    if submitted:
        with st.status("Running API call"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, tools=tools
            )
            assistant_message = response.choices[0].message
            messages.append(assistant_message)
            if assistant_message.tool_calls is not None:
                st.session_state.light_tool_calls = assistant_message.tool_calls
                for tool_call in assistant_message.tool_calls:
                    fn = globals().get(tool_call.function.name)
                    args = json.loads(tool_call.function.arguments)
                    result = fn(**args)
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": json.dumps(result),
                        }
                    )
                final_response = client.chat.completions.create(
                    model="gpt-3.5-turbo", messages=messages, tools=tools
                )
                messages.append(final_response.choices[0].message)
            st.json(messages)
        st.session_state.light_message = messages[-1].content
        st.rerun()


"""### 👀 Add Vision

You can now add an image to the context and it adds [Vision](https://platform.openai.com/docs/guides/vision) capablilities.

Be aware of the [limitations](https://platform.openai.com/docs/guides/vision/limitations)

You can use this for video frames as well! The photo is a frame from the grand finale of the [Developer Keynote at SIGNAL 2023](https://www.youtube.com/watch?v=qhNWHIZao1M).
"""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_url_for_file(image_path, image_mime_type="image/png"):
    base64_image = encode_image(image_path)
    return f"data:{image_mime_type};base64,{base64_image}"    

def chat_completion_with_vision(client, image_prompt, image_url):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": image_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            },
        ],
        max_tokens=1000,
    )
    return response


with st.form("visionary"):
    image_prompt = st.text_area("What would you like to know about this image?")
    image_file_path = os.path.join("assets", "barnburner2023.png")
    st.image(image_file_path, caption="SIGNAL 2023 - Developer Keynote")
    submitted = st.form_submit_button("Ask")
    if submitted:
        st.code(get_function_body(chat_completion_with_vision, {
            "image_prompt": image_prompt
        }))
        with st.status("Asking"):
            url = image_url_for_file(image_file_path)
            response = chat_completion_with_vision(client, image_prompt, url)
            st.markdown(response.choices[0].message.content)



"""## 🖌️ Image Generation

OpenAI's latest image generation model **DALL·E·3** is powering it's [Image Generation](https://platform.openai.com/docs/guides/images/introduction).

The [API](https://platform.openai.com/docs/api-reference/images/create) provides some additional parameters.

Be specific, it's pretty amazing.
"""


def simple_image_creation(client, user_input):
    response = client.images.generate(
        model="dall-e-3",
        prompt=user_input,
        size="1024x1024",
        n=1,
    )
    print(response.data[0].url)
    return response


with st.form("images"):
    image_user_input = st.text_area("What image should be created?")
    submitted = st.form_submit_button("Generate Image")
    if submitted:
        st.code(
            get_function_body(simple_image_creation, {"user_input": image_user_input})
        )
        with st.status("Generating image..."):
            response = simple_image_creation(client, image_user_input)
            st.image(response.data[0].url, caption=image_user_input)


def simple_text_to_speech(client, user_input, voice_choice):
    response = client.audio.speech.create(
        model="tts-1", voice=voice_choice, input=user_input
    )
    return response


"""
## 🗣️ Text to Speech

You can now use generative AI to make very realistic sounding [speech](https://platform.openai.com/docs/guides/text-to-speech).

You are currently limited to six voices using OpenAI, but this is very new. If you are looking for more, or voice cloning, check out [elevenlabs](https://elevenlabs.io/).
"""

with st.form("text-to-speech"):
    speech_user_input = st.text_area("What should the model say?")
    voice = st.selectbox(
        "Choose your voice",
        options=("alloy", "echo", "fable", "onyx", "nova", "shimmer"),
    )
    submitted = st.form_submit_button("Generate audio")
    sound = st.empty()
    if submitted:
        st.code(
            get_function_body(
                simple_text_to_speech,
                {"user_input": speech_user_input, "voice_choice": voice},
            )
        )
        response = simple_text_to_speech(client, speech_user_input, voice)
        # This will not allow for auto_play
        # example_audio = st.audio(response.content)
        b64 = base64.b64encode(response.content).decode()
        md = f"""
        <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        sound.markdown(
            md,
            unsafe_allow_html=True,
        )


"""## 👂 Speech to Text

You can get the transcriptions from the [Speech To Text](https://platform.openai.com/docs/guides/speech-to-text) audio API

This is relatively new too, you can provide a `prompt` of words that it should try to decipher in the audio. There are [workarounds](https://platform.openai.com/docs/guides/speech-to-text/prompting) to how short it is.
"""


def simple_transcription(client, audio_file, user_input=None):
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        prompt=user_input,
    )
    print(response.text)
    return response


# This is a [Streamlit plugin](https://github.com/stefanrmmr/streamlit-audio-recorder)
wav_data = st_audiorec()

if wav_data:
    with st.form("transcription"):
        submitted = st.form_submit_button("Transcribe")
        if submitted:
            # Put things in a file
            wav_file = open("tmp-recording.wav", "wb+")
            wav_file.write(wav_data)
            st.code(get_function_body(simple_transcription, {}))
            with st.status("Transcribing"):
                response = simple_transcription(client, wav_file)
                st.markdown(response.text)


"""## 📚 Adding Knowledge through Retrieval

You can actually add your own knowledge to the model.

There is an [Embeddings API](https://platform.openai.com/docs/guides/embeddings).

Make sure to check out the [Embeddings Use Cases](https://platform.openai.com/docs/guides/embeddings/use-cases)

While not necessarily a great idea for a hackathon, you could look into the [Fine-Tuning API](https://platform.openai.com/docs/guides/fine-tuning/when-to-use-fine-tuning)
"""

st.info("If this is feeling overwhelming, remember that there are lots of open source abstractions, like these from [LangChain](https://github.com/langchain-ai/langchain/blob/master/templates/docs/INDEX.md)")

""" ## 💪🚀 All together now

So where this is all headed most likely is to the next level of abstraction, the [Assistants API](https://platform.openai.com/docs/assistants/overview).

There is a [Playground](https://platform.openai.com/playground?mode=assistant) and from what it looks like is similar to "GPTs".

Note that there are also [OpenGPTs](https://github.com/langchain-ai/opengpts).
"""