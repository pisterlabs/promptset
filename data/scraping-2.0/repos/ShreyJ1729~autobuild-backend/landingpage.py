import os
import json
import modal
import openai
from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel
from utils import (
    load_openai_key,
    build_messages_from_dir,
    build_messages_from_file,
    get_component_types,
)

app = FastAPI()
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")
stub = modal.Stub("landingpage-autobuild", image=image)
mounts = [
    modal.Mount.from_local_file("./.env", remote_path="/root/.env"),
    modal.Mount.from_local_file("./utils.py", remote_path="/root/utils.py"),
    modal.Mount.from_local_dir("./prompts", remote_path="/root/prompts"),
]


@stub.webhook(method="GET", mounts=mounts)
def feature_gen(description: str):
    print("feature_gen request received: ", description)

    load_openai_key("./.env")

    messageList = build_messages_from_file(
        path="./prompts/landingpage/feature_gen.json",
        prompt=description,
    )

    print("Built messageList:", messageList)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messageList
    )

    feature_string = completion.choices[0].message.content

    print("feature_gen response:", feature_string)

    return feature_string


@stub.webhook(method="GET", mounts=mounts)
def component_gen(
    component: str,
    design: str,
    description: str,
) -> str:
    print("component-gen request received: ", description)

    load_openai_key("./.env")

    messageList = build_messages_from_dir(
        prompt_path=f"./prompts/landingpage/component_gen/prompt.json",
        data_path=f"./prompts/landingpage/component_gen/{component}",
        prompt=f"{description}. Use the following design: {design}",
    )

    print("Built messageList:", messageList)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messageList
    )

    code = completion.choices[0].message.content

    print("component-gen response:", code)

    return code


def main():
    req = {
        "description": "Vercel is the platform for frontend developers, providing  speed and reliability to shipping products through instant deployment and automatic scaling.",
        "design": "black and white, modern and minimalistic, with a teal-blue gradient background",
        "features": "Instant deployment of changes across multiple platforms, Automatic scaling to handle high amounts of traffic, Real-time collaboration features for teams, Fast and reliable global CDNs, 99.99% uptime guarantee",
    }

    pageCode = {}

    pageCode["Navbar"] = component_gen(
        component="Navbar",
        design=req["design"],
        description=f"Create a Navbar component with a logo, dark mode toggle, a link to the product page, and a link to the pricing page for the following product: {req['description']}.",
    )

    pageCode["Hero"] = component_gen(
        component="Hero",
        design=req["design"],
        description=f"Create a full-page dramatic Hero section with a title and subtitle text for the following product: {req['description']}",
    )

    pageCode["Footer"] = component_gen(
        component="Footer",
        design=req["design"],
        description=f"Create a Footer component with social links, a link to the privacy policy, and a link to the terms of service for the following product: {req['description']}.",
    )

    # write to files
    open("./Navbar.tsx", "w+").write(pageCode["Navbar"])
    open("./Hero.tsx", "w+").write(pageCode["Hero"])
    open("./Footer.tsx", "w+").write(pageCode["Footer"])


if __name__ == "__main__":
    main()


#     exit()

#     component_list = [
#         "NavBar",
#         "Hero",
#         "Details",
#         "Testimonial",
#         "Waitlist",
#         "FAQ",
#         "Footer",
#     ]

#     landingpage_code = {}

#     for component in component_list:
#         try:
#             landingpage_code[component] = component_gen(
#                 req["name"], req["description"], component
#             )
#         except Exception as e:
#             print(f"Failed to generate {component} component")
#             print(e)

#     # take all the generated code and write to files under landingpage/src
#     if not os.path.exists("./landingpage/src"):
#         os.system("npx create-react-app landingpage")
#     for component in landingpage_code:
#         code = landingpage_code[component]
#         print(f"Writing {component} component to file")
#         print(code, file=open(f"./landingpage/src/{component}.jsx", "w+"))

#     # Build App.jsx component from other components
