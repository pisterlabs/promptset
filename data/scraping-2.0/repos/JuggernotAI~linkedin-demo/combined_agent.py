# Import necessary libraries
import openai
import streamlit as st
import requests
import time
from PIL import Image
from dotenv import load_dotenv
import os
#import notion_helper
from datetime import datetime, timezone
import json
import pandas as pd
import smtplib
import notion_helper
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
 
load_dotenv()
 
openai.api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("OPENAI_ASSISTANT_DEMO")
 
client = openai
# Initialize session state variables for file IDs and chat control
if "file_id_list" not in st.session_state:
    st.session_state.file_id_list = []
 
if "start_chat" not in st.session_state:
    st.session_state.start_chat = False
 
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
 
# Set up the Streamlit page with a title and icon
st.set_page_config(page_title="Content Generator", page_icon=":speech_balloon:")
 
 
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
 
 
def post_on_twitter(twitter_post, pic=None):
    url = "https://replyrocket-backend.onrender.com/twitter/post"
    data = {"text": twitter_post}
    try:
        if pic is not None:
            with open(pic, "rb") as file:
                files = {"image": file}
                response = requests.post(url, files=files, data=data, timeout=10000)
                if response.status_code == 200:
                    return "Post successful!"
                else:
                    return f"Failed to post. Status code: {response.status_code}"
        else:
            response = requests.post(url, data=data, timeout=10000)
            if response.status_code == 200:
                return "Post successful!"
            else:
                return f"Failed to post. Status code: {response.status_code}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
 
 
def post_on_linkedin(text, pic=None):
    body = {
        "access_token": os.getenv("LINKEDIN_ACCESS_TOKEN"),
        "linkedin_id": os.getenv("LINKEDIN_ID"),
        "content": text,
    }
 
    url = "https://replyrocket-flask.onrender.com/upload"
 
    try:
        if pic is None:
            response = requests.post(url, data=body, timeout=10000)
            if response.status_code == 200:
                return "Post successful!"
            else:
                return f"Failed to post. Status code: {response.status_code}"
        else:
            with open(pic, "rb") as file:
                files = {"file": file}
                response = requests.post(url, files=files, data=body, timeout=10000)
                if response.status_code == 200:
                    return "Post successful!"
                else:
                    return f"Failed to post. Status code: {response.status_code}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
 
 
def make_post(linkedin_post, twitter_post, pic=None):
    linkedin_data = post_on_linkedin(linkedin_post, pic)
    twitter_data = post_on_twitter(twitter_post, pic)
    return linkedin_data + "\n" + twitter_data
 
 
def add_to_notion(
    linkedin_post=None,
    linkedin_post_date=None,
    twitter_post=None,
    twitter_post_date=None,
    pic=None,
):
    data = {
        "copy": {
            "title": [
                {
                    "text": {"content": ""},
                }
            ]
        },
        "image": {
            "rich_text": [
                {
                    "text": {"content": ""},
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
                "start": "",
                "end": None,
            }
        },
        "status": {
            "select": {
                "name": "Not Published",
            }
        },
        "platform": {
            "select": {
                "name": "",
            }
        },
    }
    res = ""
    datetime_format = "%B %d, %Y"
    if pic is not None:
        data["image"]["rich_text"][0]["text"]["content"] = pic
    if linkedin_post is not None:
        data["copy"]["title"][0]["text"]["content"] = linkedin_post
        data["platform"]["select"]["name"] = "Linkedin"
        dt = datetime.strptime(linkedin_post_date, datetime_format)
        formatted_date = dt.date().isoformat()
        print(formatted_date)
        data["post_date"]["date"]["start"] = formatted_date
        res = notion_helper.create_page(data)
    if twitter_post is not None:
        data["copy"]["title"][0]["text"]["content"] = twitter_post
        data["platform"]["select"]["name"] = "Twitter"
        dt = datetime.strptime(twitter_post_date, datetime_format)
        formatted_date = dt.date().isoformat()
        print(formatted_date)
        data["post_date"]["date"]["start"] = formatted_date
        res = notion_helper.create_page(data)
 
    return res
 
# Email functionality
def get_email_addresses(excel_file):
    df = pd.read_excel(excel_file)
    return df['email'].tolist()  # Replace with your column name
 
def send_email(subject, body, recipient_email, image_path):
    sender_email = "rental@flash-tech.co"  # Replace with your email
    sender_password = "Rental@009"      # Replace with your password
 
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))
 
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        message.attach(img)
 
    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()
    session.login(sender_email, sender_password)
    session.sendmail(sender_email, recipient_email, message.as_string())
    session.quit()
 
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
 
 
# Email Blasting Feature
st.sidebar.title("Email Blasting Feature")
excel_file = st.sidebar.file_uploader("Upload Excel File with Email Addresses", type=["xlsx"])
if excel_file is not None:
    email_list = get_email_addresses(excel_file)
    subject = st.sidebar.text_input("Enter the subject for the email:")
    body = st.sidebar.text_area("Enter the body content for the email:")
    image_path = st.sidebar.text_input("Enter the path of the image to attach:")
    if st.sidebar.button("Send Emails"):
        for email in email_list:
            send_email(subject, body, email, image_path)
            st.sidebar.success(f"Email sent to {email}")
 
# Only show the chat interface if the chat has been started
if st.session_state.start_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
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
            if "content" in message:
                st.markdown(message["content"])
 
    # Chat input for the user
    if prompt := st.chat_input("What is up?"):
        # Add user message to the state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
 
        # Add the user's message to the existing thread
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id, role="user", content=prompt
        )
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=assistant_id,
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
                                "linkedin_post": {
                                    "type": "string",
                                    "description": "The linkedin post text content",
                                },
                                "twitter_post": {
                                    "type": "string",
                                    "description": "The twitter post text content",
                                },
                                "image": {
                                    "type": "string",
                                    "description": "Image URL of the post generated by generate_image function",
                                },
                            },
                            "required": ["linkedin_post", "twitter_post"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "add_to_notion",
                        "description": "add data to notion",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "linkedin_post": {
                                    "type": "string",
                                    "description": "The linkedin post text content",
                                },
                                "linkedin_post_date": {
                                    "type": "string",
                                    "description": "date to post the content on linkedin in the format 'December 4, 2023'",
                                },
                                "twitter_post": {
                                    "type": "string",
                                    "description": "The twitter post text content",
                                },
                                "twitter_post_date": {
                                    "type": "string",
                                    "description": "date to post the content on twitter in the format 'December 4, 2023'",
                                },
                                "image": {
                                    "type": "string",
                                    "description": "Image URL of the post generated by generate_image function",
                                },
                            },
                            "required": [
                                "linkedin_post",
                                "linkedin_post_date",
                                "twitter_post",
                                "twitter_post_date",
                            ],
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
                    print("image generation initiated...")
                    with st.chat_message("Assistant"):
                        st.markdown("I have started the image generation process. Please wait until we come up with a visual.")
                    prompt = (
                        json.loads(tool_call.function.arguments)["prompt"]
                        + ". make sure that you do not generate images with texts in it."
                    )
                    image_url = generate_image(prompt)
                    try:
                        client.beta.threads.runs.submit_tool_outputs(
                            thread_id=st.session_state.thread_id,
                            run_id=run.id,
                            tool_outputs=[
                                {"tool_call_id": tool_call.id, "output": image_url}
                            ],
                        )
                    except Exception as e:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": e}
                        )
                        with st.chat_message("assistant"):
                            st.markdown(e)
                elif tool_call.function.name == "make_post":
                    print("make post initiated...")
                    linkedin_post = json.loads(tool_call.function.arguments).get(
                        "linkedin_post", None
                    )
                    twitter_post = json.loads(tool_call.function.arguments).get(
                        "twitter_post", None
                    )
                    pic = json.loads(tool_call.function.arguments).get("image", None)
                    data = ""
                    if pic != None:
                        path = os.path.join(
                            "./dalle", str(round(time.time() * 1000)) + ".png"
                        )
                        Image.open(
                            requests.get(pic, stream=True, timeout=10000).raw
                        ).save(path)
                        data = make_post(linkedin_post, twitter_post, path)
                    else:
                        data = make_post(linkedin_post, twitter_post)
                    try:
                        client.beta.threads.runs.submit_tool_outputs(
                            thread_id=st.session_state.thread_id,
                            run_id=run.id,
                            tool_outputs=[
                                {"tool_call_id": tool_call.id, "output": data}
                            ],
                        )
                    except Exception as e:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": e}
                        )
                        with st.chat_message("assistant"):
                            st.markdown(e)
                elif tool_call.function.name == "add_to_notion":
                    print("add to notion initiated...")
                    linkedin_post = json.loads(tool_call.function.arguments).get(
                        "linkedin_post", None
                    )
                    twitter_post = json.loads(tool_call.function.arguments).get(
                        "twitter_post", None
                    )
                    twitter_post_date = json.loads(tool_call.function.arguments).get(
                        "twitter_post_date", None
                    )
                    linkedin_post_date = json.loads(tool_call.function.arguments).get(
                        "linkedin_post_date", None
                    )
                    pic = json.loads(tool_call.function.arguments).get("image", None)
                    data = ""
                    if pic is not None:
                        path = os.path.join(
                            "./dalle", str(round(time.time() * 1000)) + ".png"
                        )
                        Image.open(
                            requests.get(pic, stream=True, timeout=10000).raw
                        ).save(path)
 
                        data = add_to_notion(
                            linkedin_post,
                            linkedin_post_date,
                            twitter_post,
                            twitter_post_date,
                            path,
                        )
                    else:
                        data = add_to_notion(
                            linkedin_post,
                            linkedin_post_date,
                            twitter_post,
                            twitter_post_date,
                        )
                    try:
                        client.beta.threads.runs.submit_tool_outputs(
                            thread_id=st.session_state.thread_id,
                            run_id=run.id,
                            tool_outputs=[
                                {"tool_call_id": tool_call.id, "output": data}
                            ],
                        )
                    except Exception as e:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": e}
                        )
                        with st.chat_message("assistant"):
                            st.markdown(e)
 
        messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread_id
        )
        for message in [
            m for m in messages if m.run_id == run.id and m.role == "assistant"
        ]:
            full_response = process_message_with_citations(message)
            if st.session_state.image_count > st.session_state.image_count_temp:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "image": st.session_state.image_paths[-1],
                    }
                )
                with st.chat_message("assistant"):
                    st.image(st.session_state.image_paths[-1])
                st.session_state.image_count_temp += 1
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                with st.chat_message("assistant"):
                    st.markdown(full_response, unsafe_allow_html=True)
else:
    st.write("Please click 'Start Chat' to begin the conversation.")
 
 