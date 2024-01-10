import autogen
import panel as pn
import openai
import os
import time
import importlib

import asyncio
from supabase import create_client, Client

# dotenv
from dotenv import load_dotenv

load_dotenv()

# load from env file
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Configuration for GPT-4
MODEL = "gpt-4-1106-preview"
API_KEY = os.getenv("OPENAI_API_KEY")
config_list = [{"model": MODEL, "api_key": API_KEY}]
gpt4_config = {"config_list": config_list, "temperature": 0, "seed": 53}

# Registration Page Components
email_input = pn.widgets.TextInput(name="Email", placeholder="Enter Email")
api_key_input = pn.widgets.PasswordInput(name="API Key", placeholder="Enter API Key")
model_select = pn.widgets.Select(name="Model", options=["gpt-4-1106-preview"])
register_button = pn.widgets.Button(name="Register", button_type="primary")
custom_style_hyperlink_button = {
    "background": "#f5f5f5",
    "border": "none",
    "color": "blue",
    "text-decoration": "underline",
    "font-size": "12px",
}
go_to_login_button = pn.widgets.Button(
    name="Go to Login", styles=custom_style_hyperlink_button
)

# Login Page Components
email_login_input = pn.widgets.TextInput(name="Email", placeholder="Enter Email")
login_button = pn.widgets.Button(name="Login", button_type="primary")

# Main Area for Dynamic Content
main_area = pn.Column()


# Page Switch Function
def switch_page(target_page, user_api_key=None, user_model=None):
    if target_page == "login":
        main_area.objects = [create_login_page()]
    elif target_page == "register":
        main_area.objects = [create_registration_page()]
    elif target_page == "chat":
        main_area.objects = [create_chat_interface(user_api_key, user_model)]


# Callback Functions
def register_callback(event):
    email = email_input.value
    api_key = api_key_input.value

    def encrypt_api_key(api_key):
        """
        Encrypts the API key using a simple Caesar cipher
        """
        encrypted_api_key = ""
        for char in api_key:
            encrypted_api_key += chr(ord(char) + 1)
        return encrypted_api_key

    api_key = encrypt_api_key(api_key)
    model = model_select.value
    response = (
        supabase.table("user_preferences")
        .insert({"email": email, "api_key": api_key, "model": model})
        .execute()
    )
    switch_page("login")


def go_to_login_callback(event):
    switch_page("login")


def login_callback(event):
    email = email_login_input.value
    user_data = (
        supabase.table("user_preferences")
        .select("api_key, model")
        .eq("email", email)
        .execute()
    )
    user_data = user_data.data
    if len(user_data):
        user_data = user_data[0]
        user_api_key = user_data["api_key"]

        def decrypt_api_key(api_key):
            """
            Decrypts the API key using a simple Caesar cipher
            """
            decrypted_api_key = ""
            for char in api_key:
                decrypted_api_key += chr(ord(char) - 1)
            return decrypted_api_key

        user_api_key = decrypt_api_key(user_api_key)
        user_model = user_data["model"]
        switch_page("chat", user_api_key, user_model)
    else:
        print("Login failed")


# Page Creation Functions
def create_registration_page():
    registration_layout = pn.Column(
        pn.layout.HSpacer(),  # Horizontal spacer for centering
        pn.Column(  # Nested column for the form
            pn.Spacer(height=100),  # Vertical spacer for centering from top
            pn.Spacer(width=200),
            email_input,
            api_key_input,
            model_select,
            register_button,
            go_to_login_button,
            align="center",  # Align items in the column to the center
            sizing_mode="fixed",  # Use fixed sizing mode for better control
            width=400,  # Set a fixed width for the form
            background="#f5f5f5",
            margin=(0, 200, 0, 200),
        ),
        pn.layout.HSpacer(),  # Horizontal spacer for centering
        align="center",  # Align the whole layout to the center
        sizing_mode="stretch_both",  # Stretch to fill the available space
    )
    register_button.on_click(register_callback)
    go_to_login_button.on_click(go_to_login_callback)
    return registration_layout


def create_login_page():
    login_page = pn.Column(email_login_input, login_button)
    login_button.on_click(login_callback)
    return login_page


def create_chat_interface(user_api_key, user_model):
    global input_future  # Declare input_future as a global variable
    input_future = None  # Initialize input_future

    # Set environment variable for OpenAI API key
    os.environ["OPENAI_API_KEY"] = user_api_key

    # Configuration for the selected GPT model
    gpt4_config = {"config_list": [{"model": user_model}], "temperature": 0, "seed": 53}

    def register_agent_replies(agents, groupchat, manager):
        for agent in agents:
            agent.register_reply(
                [autogen.Agent, None],
                reply_func=lambda recipient, messages, sender, config: print_messages(
                    recipient, messages, sender, config, groupchat, manager
                ),
                config={"callback": None},
            )

    def print_messages(recipient, messages, sender, config, groupchat, manager):
        chat_interface.send(
            messages[-1]["content"],
            user=messages[-1]["name"],
            respond=False,
        )
        print(
            f"Messages from: {sender.name} sent to: {recipient.name} | num messages: {len(messages)} | message: {messages[-1]}"
        )
        if "name" in messages[-1]:
            chat_interface.send(
                messages[-1]["content"],
                user=messages[-1]["name"],
                respond=False,
            )
        else:
            chat_interface.send(
                messages[-1]["content"], user="SecretGuy", respond=False
            )
        return False, None

    async def chat_callback(
        contents: str, user: str, instance: pn.widgets.Terminal, user_proxy, manager
    ):
        global input_future  # Ensure that you're using the global variable
        if not input_future or input_future.done():
            asyncio.create_task(delayed_initiate_chat(user_proxy, manager, contents))
        else:
            if input_future and not input_future.done():
                input_future.set_result(contents)

    async def delayed_initiate_chat(user_proxy, manager, contents):
        # await asyncio.sleep(2)
        await user_proxy.a_initiate_chat(manager, message=contents)

    # Define a conversable agent class
    class MyConversableAgent(autogen.ConversableAgent):
        async def a_get_human_input(self, prompt: str) -> str:
            global input_future
            chat_interface.send(prompt, user="System", respond=False)
            if input_future is None or input_future.done():
                input_future = asyncio.Future()
            await input_future
            input_value = input_future.result()
            input_future = None
            return input_value

    # Create agents
    user_proxy = MyConversableAgent(
        name="Admin",
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("exit"),
        system_message="""A human admin. Interact with the planner to discuss the plan.""",
        code_execution_config=False,
        human_input_mode="ALWAYS",
        llm_config=gpt4_config,
    )

    import sys

    sys.path.append(os.getcwd())

    agents = importlib.import_module("agents")

    returned_agents = agents.agents_defination(gpt4_config)

    # Group chat setup
    groupchat = autogen.GroupChat(
        agents=[user_proxy, *returned_agents],
        messages=[],
        max_round=20,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

    # Register replies for each agent
    register_agent_replies(
        [user_proxy, *returned_agents],
        groupchat,
        manager,
    )

    # Chat Interface
    chat_interface = pn.chat.ChatInterface(
        callback=lambda contents, user, instance: asyncio.create_task(
            chat_callback(contents, user, instance, user_proxy, manager)
        )
    )
    chat_interface.send("Send a message!", user="System", respond=False)

    return chat_interface


# Initialize with Registration Page
main_area.objects = [create_registration_page()]

# Serve the Application
pn.serve(main_area)
