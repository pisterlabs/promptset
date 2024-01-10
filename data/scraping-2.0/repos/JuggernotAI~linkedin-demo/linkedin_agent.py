# Import necessary libraries
import openai
import streamlit as st
import requests
import time
from PIL import Image
from dotenv import load_dotenv
import os
import re
import notion_helper
import json
from datetime import datetime, timezone

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
# Set your OpenAI Assistant ID here
assistant_id = os.getenv("OPENAI_LINKEDIN_ASSISTANT_ID")
instruction = """
As the 'LinkedIn Content Specialist' at AgentGPT, your primary role is to assist users in crafting professional and engaging content for LinkedIn. This involves creating comprehensive, long-form LinkedIn posts up to the 3000-character limit, tailored to the platform's standards. You'll be focusing on a variety of themes such as business insights, industry news, thought leadership, career advice, or inspirational stories, depending on user preferences.

Each request you handle will involve generating a single, detailed post variant. Your content creation will be strictly text-based unless a user specifically requests the incorporation of visual elements. These posts should be structured to be relevant, engaging, and suitable for the LinkedIn audience. Adapt your responses to meet the specific context of the user's needs, aiming to create LinkedIn-specific content that ensures user satisfaction. Users will provide clear, detailed input about their desired content and may offer feedback for refinement.

Key Responsibilities:

- Generate a single detailed LinkedIn Post, up to 3000 characters, covering a wide range of professional themes.
- Focus on text content creation, engaging only in visual content creation when specifically asked by the user.
- Utilize the generate_image function only upon explicit user request for an image to complement their LinkedIn post.
- Avoid adding any placeholder content in the post or any image credits such as "[Image: Courtesy of OpenAI's DALL·E]".
- Utilize the make_post function for posting the content to LinkedIn as per user requests."""

client = openai
# Initialize session state variables for file IDs and chat control
if "file_id_list" not in st.session_state:
    st.session_state.file_id_list = []

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Set up the Streamlit page with a title and icon
st.set_page_config(page_title="Linkedin Content Creator", page_icon=":speech_balloon:")


def extract_text(text):
    matches = re.findall(r'"([^"]*)"', text)
    st.session_state.extracted_text = matches[0] if matches else None
    print("Extracted text:", st.session_state.extracted_text)


def process_message_with_citations(message):
    return message.content[0].text.value


def generate_image(prompt, size="1024x1024"):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        n=1,
        quality="standard",
    )
    path = os.path.join("./dalle", str(round(time.time() * 1000)) + ".png")
    image_url = response.data[0].url
    Image.open(requests.get(image_url, stream=True).raw).save(path)
    st.session_state.image_paths.append(path)
    st.session_state.image_count += 1
    return image_url


def linkedin_post():
    headers = {
        "Content-Type": "application/json",
    }
    if len(st.session_state.image_paths) > 0:
        data = {
            "access_token": os.getenv("LINKEDIN_ACCESS_TOKEN"),
            "linkedin_id": os.getenv("LINKEDIN_ID"),
            "content": st.session_state.extracted_text,
        }
        url = "https://replyrocket-flask.onrender.com/upload"
        try:
            with open(st.session_state.image_paths[-1], "rb") as file:
                files = {"file": file}
                response = requests.post(url, files=files, data=data, timeout=10000)
                if response.status_code == 200:
                    return "Post successful!"
                else:
                    return f"Failed to post. Status code: {response.status_code}"
        except Exception as e:
            return f"An error occurred: {str(e)}"
    else:
        body = {
            "access_token": os.getenv("LINKEDIN_ACCESS_TOKEN"),
            "linkedin_id": os.getenv("LINKEDIN_ID"),
            "content": st.session_state.extracted_text,
        }
        url = "https://replyrocket-flask.onrender.com/post"
        try:
            response = requests.post(url, json=body, headers=headers, timeout=10000)
            if response.status_code == 200:
                return "Post successful!"
            else:
                return f"Failed to post. Status code: {response.status_code}"
        except Exception as e:
            return f"An error occurred: {str(e)}"


def make_post(text, pic=None):
    headers = {
        "Content-Type": "application/json",
    }

    body = {
        "access_token": os.getenv("LINKEDIN_ACCESS_TOKEN"),
        "linkedin_id": os.getenv("LINKEDIN_ID"),
        "content": text,
    }

    if pic is None:
        url = "https://replyrocket-flask.onrender.com/post"
        try:
            response = requests.post(url, json=body, headers=headers, timeout=10000)
            if response.status_code == 200:
                return "Post successful!"
            else:
                return f"Failed to post. Status code: {response.status_code}"
        except Exception as e:
            return f"An error occurred: {str(e)}"
    else:
        url = "https://replyrocket-flask.onrender.com/upload"
        try:
            path = os.path.join("./dalle", str(round(time.time() * 1000)) + ".png")
            Image.open(requests.get(pic, stream=True).raw).save(path)
            with open(path, "rb") as file:
                files = {"file": file}
                response = requests.post(url, files=files, data=body, timeout=10000)
                if response.status_code == 200:
                    return "Post successful!"
                else:
                    return f"Failed to post. Status code: {response.status_code}"
        except Exception as e:
            return f"An error occurred: {str(e)}"


# Start Chat Button
if st.sidebar.button("Start Chat"):
    st.session_state.start_chat = True
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id
    st.write("Thread ID: ", thread.id)

# Main chat interface setup
st.title("Agent Baani")
st.write(
    """As an AI copilot  for making posts on Social Media, I will assist you with making an engaging copy.

*Made By Juggernot.ai* """
)

# Only show the chat interface if the chat has been started
if st.session_state.start_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = ""
    if "image_count_temp" not in st.session_state:
        st.session_state.image_count_temp = 0
    if "image_paths" not in st.session_state:
        st.session_state.image_paths = []
    if "image_count" not in st.session_state:
        st.session_state.image_count = 0

    # Display existing messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "image" in message:
                st.image(message["image"])
            st.markdown(message["content"])

    # Chat input for the user
    if prompt := st.chat_input("What is up?"):
        # Add user message to the state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if prompt.startswith("!post"):
            res = linkedin_post()
            st.session_state.messages.append({"role": "assistant", "content": res})
            with st.chat_message("assistant"):
                st.markdown(res)
        elif prompt.startswith("!delete"):
            pages = notion_helper.get_pages()
            data = []
            for page in pages:
                page_id = page["id"]
                props = page["properties"]
                copy = props["copy"]["title"][0]["text"]["content"]
                image = props["image"]["rich_text"][0]["text"]["content"]
                created_at = props["created_at"]["date"]["start"]
                post_date = props["post_date"]["date"]["start"]
                status = props["status"]["select"]["name"]
                platform = props["platform"]["multi_select"][0]["name"]
                data.append(
                    {
                        "id": page_id,
                        "copy": copy,
                        "image": image,
                        "created_at": created_at,
                        "post_date": post_date,
                        "status": status,
                        "platform": platform,
                    }
                )
            st.markdown(json.dumps(data, indent=4))
        elif prompt.startswith("!database"):
            data = {
                "copy": {
                    "title": [
                        {
                            "text": {"content": "tweet 3.0"},
                        }
                    ]
                },
                "image": {
                    "rich_text": [
                        {
                            "text": {
                                "content": "https://media.wired.com/photos/5b899992404e112d2df1e94e/master/pass/trash2-01.jpg"
                            },
                        }
                    ]
                },
                "created_at": {
                    "date": {
                        "start": datetime.now(timezone.utc).date().isoformat(),
                        "end": None,
                    }
                },
                "post_date": {
                    "date": {
                        "start": datetime.now(timezone.utc).date().isoformat(),
                        "end": None,
                    }
                },
                "status": {
                    "select": {
                        "name": "Not Published",
                    }
                },
                "platform": {
                    "multi_select": [
                        {
                            "name": "Twitter",
                        }
                    ]
                },
            }
            notion_helper.create_page(data)
            st.markdown("added")
        else:
            # Add the user's message to the existing thread
            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id, role="user", content=prompt
            )
            run = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=assistant_id,
                instructions=instruction,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "generate_image",
                            "description": "generate image by Dall-e 3",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "prompt": {
                                        "type": "string",
                                        "description": "The prompt to generate image",
                                    },
                                    "size": {
                                        "type": "string",
                                        "enum": ["1024x1024", "other_sizes"],
                                    },
                                },
                                "required": ["prompt"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "make_post",
                            "description": "make a post to linkedin",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "post": {
                                        "type": "string",
                                        "description": "The linkedin post text content",
                                    },
                                    "image": {
                                        "type": "string",
                                        "description": "Image for the linkedin post generated by generate_image function",
                                    },
                                },
                                "required": ["post"],
                            },
                        },
                    },
                ],
            )

            while run.status != "completed":
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread_id, run_id=run.id
                )
                if run.status == "requires_action":
                    tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
                    if tool_call.function.name == "generate_image":
                        prompt = (
                            json.loads(tool_call.function.arguments)["prompt"]
                            + ". make sure that you do not generate images with texts in it."
                        )
                        image_url = generate_image(prompt)
                        client.beta.threads.runs.submit_tool_outputs(
                            thread_id=st.session_state.thread_id,
                            run_id=run.id,
                            tool_outputs=[
                                {"tool_call_id": tool_call.id, "output": image_url}
                            ],
                        )
                    elif tool_call.function.name == "make_post":
                        print("make post initiated...")
                        text = json.loads(tool_call.function.arguments)["post"]
                        pic = json.loads(tool_call.function.arguments)["image"]
                        data = ""
                        if pic != None:
                            data = make_post(text, pic)
                        else:
                            data = make_post(text)
                        client.beta.threads.runs.submit_tool_outputs(
                            thread_id=st.session_state.thread_id,
                            run_id=run.id,
                            tool_outputs=[
                                {"tool_call_id": tool_call.id, "output": data}
                            ],
                        )

            messages = client.beta.threads.messages.list(
                thread_id=st.session_state.thread_id
            )
            for message in [
                m for m in messages if m.run_id == run.id and m.role == "assistant"
            ]:
                full_response = process_message_with_citations(message)
                extract_text(full_response)
                if st.session_state.image_count > st.session_state.image_count_temp:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "image": st.session_state.image_paths[-1],
                        }
                    )
                    with st.chat_message("assistant"):
                        st.image(st.session_state.image_paths[-1])
                        st.markdown(full_response, unsafe_allow_html=True)
                    st.session_state.image_count_temp += 1
                else:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                    with st.chat_message("assistant"):
                        st.markdown(full_response, unsafe_allow_html=True)
else:
    st.write("Please click 'Start Chat' to begin the conversation.")
