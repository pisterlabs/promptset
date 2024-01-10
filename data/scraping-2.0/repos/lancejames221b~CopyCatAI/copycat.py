import pyperclip
import PySimpleGUI as sg
import os
from PIL import ImageGrab, Image
from pathlib import Path
import traceback
import configparser
import webbrowser
import json
import openai

from extract import *
from notification import *
from gptplus import *
from prompt_ui import *
from splash import *
import os
import shutil
from pathlib import Path

home_dir = os.path.expanduser("~")
bundle_dir = os.path.join(home_dir, "Library", "Application Support", "CopyCat")
Path(bundle_dir).mkdir(parents=True, exist_ok=True)
models_path = os.path.join(bundle_dir, "models.json")

# Define the source file
source_file = "./models.json"

# Check if the file exists at the destination
if not os.path.exists(models_path):
    # Copy the file
    shutil.copy(source_file, models_path)


def reset_costs(config_path):
    global CONFIG
    CONFIG = load_config(config_path)
    CONFIG.set("OpenAI", "total_costs", str(0))  # Reset total costs to 0
    save_config(config_path, CONFIG)


def copy_files(filename, binary=False):
    if getattr(sys, "frozen", False):
        app_config_dir = sys._MEIPASS
    else:
        app_config_dir = os.path.dirname(os.path.abspath(__file__))

    bundled_config_file = os.path.join(app_config_dir, filename)
    destination_file = os.path.join(bundle_dir, filename)

    if not binary:
        temp_config = open(bundled_config_file, "r").read()
        with open(destination_file, "w") as f:
            f.write(temp_config)
    else:
        temp_config = open(bundled_config_file, "rb").read()
        with open(destination_file, "wb") as f:
            f.write(temp_config)

    os.chmod(destination_file, 0o600)
    print("created", destination_file)


logo_path = os.path.join(bundle_dir, "logo.png")
config_path = os.path.join(bundle_dir, "config.ini")
memory_path = os.path.join(bundle_dir, "memory.json")
first_time = os.path.join(bundle_dir, "first_time.txt")

if not os.path.exists(config_path):  # If the config file doesn't exist
    copy_files("config.ini")
if not os.path.exists(logo_path):  # If the logo doesn't exist
    copy_files("logo.png", binary=True)

create_splash_screen()


def if_first_time():
    if not os.path.exists(first_time):
        with open(first_time, "w") as f:
            f.write("1")
        return True
    else:
        return False


def make_memory_file(filepath=memory_path):
    with open(filepath, "w") as f:
        json.dump({}, f, indent=4)


if not os.path.exists(memory_path):
    make_memory_file()


def load_config(filepath):
    config = configparser.ConfigParser(strict=False, interpolation=None)
    config.read(filepath)
    return config


def save_config(filepath, config_dict):
    config = configparser.ConfigParser(strict=False, interpolation=None)
    for section, options in config_dict.items():
        config[section] = options
    with open(filepath, "w") as config_file:
        config.write(config_file)


def is_api_key_empty(api_key):
    return len(api_key.strip()) == 0


def is_notion_token_empty(token, space_id):
    print("is_notion_token_empty")
    print(token, space_id)
    return len(token.strip()) == 0 or len(space_id.strip()) == 0


CONFIG = load_config(config_path)


def settings_window(main_window=None):
    global CONFIG
    with open(models_path, "r") as f:
        models = json.load(f)
    model_info = "\n".join(
        [f"{model}: {info['token_size']} max tokens" for model, info in models.items()]
    )

    max_tokens = CONFIG.get("OpenAI", "max_tokens")
    if max_tokens == "0":
        max_tokens = None
    else:
        max_tokens = int(max_tokens)

    model = CONFIG.get("OpenAI", "model")

    if model in models:
        max_range = models[model]["token_size"]
    else:
        raise ValueError(f"Invalid model name: {model}")

    layout = [
        [
            sg.Text("OpenAI API Key:"),
            sg.InputText(default_text=CONFIG.get("OpenAI", "api_key"), key="api_key"),
        ],
        [
            sg.Text("OpenAI Temperature (default 0.8):"),
            sg.Slider(
                range=(0, 1),
                default_value=float(CONFIG.get("OpenAI", "temperature")),
                resolution=0.1,
                orientation="h",
                size=(20, 15),
                key="temperature",
            ),
        ],
        [
            sg.Text(f"Max Tokens (default None. {model_info}):"),
            sg.Spin(
                values=[i for i in range(0, 128001)],
                initial_value=max_tokens,
                size=(6, 1),
                key="max_tokens",
            ),
        ],
        [sg.Button("Save"), sg.Button("Cancel")],
    ]
    if main_window:
        main_window.Hide()
    swindow = sg.Window("Settings", layout, finalize=True)
    swindow.bring_to_front()
    while True:
        event, values = swindow.read()

        if event == "Save":
            CONFIG.set("OpenAI", "api_key", str(values["api_key"]))
            CONFIG.set("OpenAI", "temperature", str(values["temperature"]))
            save_config(config_path, CONFIG)

            sg.popup("Settings saved!", keep_on_top=True)
            swindow.close()
            break
        elif event == "Cancel" or event == sg.WIN_CLOSED:
            swindow.close()
            break

    if main_window:
        main_window.UnHide()
        main_window.bring_to_front()
    return


PROMPT = False
SKIP = False
DEBUG = False
TEST = False
window_location = (None, None)
include_urls = CONFIG.getboolean("GUI", "include_urls")
mem_on_off = CONFIG.getboolean("GUI", "mem_on_off")
TOPIC = CONFIG.get("GUI", "topic")
user = CONFIG.get("GUI", "user")
codemode = CONFIG.getboolean("GUI", "codemode")
costs = str(float(CONFIG.get("OpenAI", "costs")))
total_costs = str(float(CONFIG.get("OpenAI", "total_costs")))
total_tokens = str(float(CONFIG.get("OpenAI", "total_tokens")))
model = CONFIG.get("OpenAI", "model")
api_key = CONFIG.get("OpenAI", "api_key")
temperature = CONFIG.get("OpenAI", "temperature")
max_tokens = CONFIG.get("OpenAI", "max_tokens", fallback=None)
if max_tokens == "None":
    max_tokens = None
else:
    max_tokens = int(max_tokens)
openai.api_key = api_key

if if_first_time():
    webbrowser.open_new_tab(
        "https://313372600.notion.site/CopyCat-AI-Instructions-f94df67d0f3e47c89bd93810e38fb272"
    )
    settings_window()


def prompt_user(clip, img=False):
    global CONFIG
    global PROMPT
    global SKIP
    global mem_on_off
    global TOPIC
    global codemode
    global model
    global costs
    global total_costs
    global total_tokens
    global model
    global temperature
    global max_tokens
    global api_key
    global include_urls
    global window_location
    CONFIG = load_config(config_path)
    PROMPT = False
    SKIP = False
    DEBUG = True
    with open(models_path, "r") as f:
        models = json.load(f)

    include_urls = CONFIG.getboolean("GUI", "include_urls")
    mem_on_off = CONFIG.getboolean("GUI", "mem_on_off")
    TOPIC = CONFIG.get("GUI", "topic")
    user = CONFIG.get("GUI", "user")
    codemode = CONFIG.getboolean("GUI", "codemode")
    costs = str(float(CONFIG.get("OpenAI", "costs")))
    total_costs = str(float(CONFIG.get("OpenAI", "total_costs")))
    total_tokens = str(float(CONFIG.get("OpenAI", "total_tokens")))
    model = CONFIG.get("OpenAI", "model")
    api_key = CONFIG.get("OpenAI", "api_key")
    temperature = CONFIG.get("OpenAI", "temperature")
    max_tokens = CONFIG.get("OpenAI", "max_tokens", fallback=None)
    model_names = list(models.keys())
    model_info = ", ".join(
        [f"{model}: {info['token_size']} max tokens" for model, info in models.items()]
    )
    if max_tokens == "0":
        max_tokens = None
    else:
        max_tokens = int(max_tokens)
    openai.api_key = api_key
    if is_api_key_empty(api_key) and is_notion_token_empty(
        CONFIG.get("NotionAI", "token_v2"), CONFIG.get("NotionAI", "space_id")
    ):
        settings_window()
        api_key = CONFIG.get("OpenAI", "api_key")
        openai.api_key = api_key

    openai_memory = OpenAIMemory(memory_path, config_path)
    input_text = ""
    PROMPT = True
    if DEBUG:
        print("Loading Prompt")

    try:
        layout = [
            [
                sg.Menu(
                    [["Settings", ["Preferences", "Prompt Manager", "About", "Help"]]],
                    key="-MENU-",
                )
            ],
            [
                sg.Text(
                    "What are your orders? (Press enter to submit or escape to cancel)"
                )
            ],
            [
                sg.Input(
                    key="input", tooltip="Press enter key to submit", do_not_clear=False
                ),
                sg.Button(
                    "OK",
                    tooltip="Press enter key to submit",
                    visible=False,
                    bind_return_key=True,
                    key="-RETURN-",
                ),
                sg.Button(
                    "Cancel",
                    tooltip="Press escape key to cancel",
                    visible=False,
                    bind_return_key=True,
                    key="-ESCAPE-",
                ),
                sg.Text("Select Model:", tooltip="Select Model"),
                sg.Combo(
                    model_names,
                    default_value=model,
                    readonly=True,
                    key="-MODEL-",
                    enable_events=True,
                ),
            ],
            [
                sg.Column(
                    [
                        [
                            sg.Checkbox(
                                "Include Url Pages",
                                key="-URLS-",
                                default=include_urls,
                                tooltip="Include Url Pages",
                            ),
                        ],
                        [
                            sg.Checkbox(
                                "Memory On/Off",
                                default=mem_on_off,
                                key="memory_on_off",
                                tooltip="Handy if you're editing docs",
                            ),
                        ],
                        [
                            sg.Checkbox(
                                "Coding Mode",
                                default=codemode,
                                key="code",
                                tooltip="Coding Mode",
                            ),
                        ],
                        [
                            sg.Checkbox(
                                "Preview",
                                default=False,
                                key="-PREVIEW-",
                                tooltip="Preview Mode",
                                enable_events=True,
                            ),
                        ],
                    ],
                    pad=(0, 0),
                ),
                sg.VerticalSeparator(),
                sg.Frame(
                    title="API Info",
                    layout=[
                        [
                            sg.Text("Last Request Costs: "),
                            sg.Text("0", key="lrcosts", size=(10, 1)),
                        ],
                        [
                            sg.Text("Total Requests Costs: "),
                            sg.Text("0", key="total", size=(10, 1)),
                        ],
                        [
                            sg.Text("Temperature: "),
                            sg.Text(temperature, key="temperature", size=(10, 1)),
                        ],
                        [
                            sg.Text("Last Request Tokens: "),
                            sg.Text(max_tokens, key="total_tokens", size=(10, 1)),
                        ],
                    ],
                    relief=sg.RELIEF_SUNKEN,
                    border_width=2,
                ),
                sg.VerticalSeparator(),
                sg.Frame(
                    title="",
                    layout=[
                        [sg.Image(logo_path, size=(100, 100))],
                    ],
                    relief=sg.RELIEF_SUNKEN,
                    border_width=2,
                ),
            ],
            [
                sg.Button(
                    "Clear Memory",
                    key="Clear Memory",
                    tooltip="Clears the memory and starts fresh",
                    enable_events=True,
                ),
                sg.Button(
                    "Reset Costs",
                    key="Reset Costs",
                    tooltip="Resets the total request costs back to 0",
                    enable_events=True,
                ),
            ],
            [
                sg.Text(
                    "New System Prompt: ",
                    tooltip="Topic to use for the prompt. Default is general",
                ),
                sg.Input(
                    key="topic",
                    tooltip="Press enter key to submit",
                    do_not_clear=False,
                    default_text=TOPIC,
                    size=(54, 1),
                ),
            ],
            [
                sg.Text(
                    "Existing System Prompt: ",
                    tooltip="Topic to use for the prompt. Default is general",
                ),
                sg.Combo(
                    openai_memory.get_memory_keys(),
                    default_value=TOPIC,
                    key="-COMBO-",
                    tooltip="Press enter key to submit",
                    readonly=True,
                    enable_events=True,
                    size=(28, 1),
                ),
                sg.Button(
                    "Delete System Prompt",
                    key="Delete Topic",
                    tooltip="Deletes the topic",
                    enable_events=True,
                ),
            ],
            [sg.Multiline(size=(75, 10), key="-PREVIEW-ML-", visible=False)],
        ]

        window = sg.Window(
            "CopyCat AI",
            layout,
            keep_on_top=True,
            auto_close=True,
            auto_close_duration=120,
            location=window_location,
            finalize=True,
        )

        window.bind("<Escape>", "-ESCAPE-")
        window.bind("<Return>", "OK")

        window.bring_to_front()
        if DEBUG:
            print("Prompt Loaded", window)

        window["lrcosts"].update(str(float(costs)))
        window["total"].update(str(float(total_costs)))
        window["total_tokens"].update(str(total_tokens))

        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, "Cancel", "-ESCAPE-"):
                break  # Ensure that all conditions leading to a window closure are handled

            TOPIC = values["-COMBO-"]
            window_location = window.current_location()
            if event == sg.TIMEOUT_KEY:
                window.close()
                return
            if event == sg.WIN_CLOSED or event == None:  # Check for event == WIN_CLOSED
                return

            elif event == "Prompt Manager":
                prompt_manager = PromptManager(memory_path)
                prompt_manager.prompt_manager(main_window=window)
                openai_memory.load_memory()
                window["-COMBO-"].update(openai_memory.get_memory_keys())
                window.refresh()

            elif event == "About":
                sg.popup(
                    "Title: CopyCat AI\n\nCopyright: Nobility AI, 2023\n\nWebsite: https://nobilityai.com\n\nEmail: copycat@nobilityai.com\n\n",
                    keep_on_top=True,
                )

            elif event == "Preferences":
                settings_window(main_window=window)
                temperature = CONFIG.get("OpenAI", "temperature")
                max_tokens = CONFIG.get("OpenAI", "max_tokens")
                if max_tokens == "0":
                    max_tokens = None
                else:
                    max_tokens = int(max_tokens)
                openai.api_key = CONFIG.get("OpenAI", "api_key")
                window["temperature"].update(temperature)
                window["total_tokens"].update(max_tokens)
                window.refresh()
                # api_key = CONFIG.get("OpenAI", "api_key")
                # openai.api_key = api_key
                # temperature = CONFIG.get("OpenAI", "temperature")
                # max_tokens = CONFIG.get("OpenAI", "max_tokens")

            elif event == "Help":
                webbrowser.open("https://github.com/lancejames221b/CopyCatAI")

            if event == "Clear Memory":
                if DEBUG:
                    print("Clearing memory")

                openai_memory.clear_memory(TOPIC)
                window.refresh()
                # memory(user=user, topic=TOPIC, reset=True)
                TOPIC = values["-COMBO-"]
                if DEBUG:
                    print(TOPIC)
                display_notification(
                    "Topic Memory Cleared",
                    TOPIC + " memory has been cleared",
                    img_success,
                    5000,
                    use_fade_in=False,
                    location=(0, 0),
                )

            if event == "Delete Topic":
                if DEBUG:
                    print("DELETING TOPIC", values["-COMBO-"])
                if (
                    values["topic"] != ""
                    and values["topic"]
                    != "You are an AI assistant helping with knowledge and code."
                ):
                    TOPIC = values[
                        "topic"
                    ]  # use values['topic'] instead of values['-COMBO-']
                    if DEBUG:
                        print("Deleting topic")
                    # memory(user=user, topic=TOPIC, delete=True)
                    openai_memory.clear_memory(TOPIC)
                    CONFIG["GUI"][
                        "topic"
                    ] = "You are an AI assistant helping with knowledge and code."
                    window["-COMBO-"].update(
                        values=openai_memory.get_memory_keys(),
                        value="You are an AI assistant helping with knowledge and code.",
                    )
                    TOPIC = values["topic"]
                    save_config(config_path, CONFIG)
                    # event, values = window.read()
                    window.refresh()
                    display_notification(
                        "Topic Deleted",
                        TOPIC + " has been deleted",
                        img_success,
                        5000,
                        use_fade_in=False,
                        location=(0, 0),
                    )

                else:
                    if DEBUG:
                        print("Cannot delete default topic")
                    window["-COMBO-"].update(
                        values=openai_memory.get_memory_keys(),
                        value="You are an AI assistant helping with knowledge and code.",
                    )
                    TOPIC = values["topic"]
                    # memory(user=user, topic="general", reset=True)
                    openai_memory.clear_memory(
                        "You are an AI assistant helping with knowledge and code."
                    )
                    window.refresh()
                    display_notification(
                        "Default Topic Reset",
                        "Can't Delete Default Topic",
                        img_success,
                        5000,
                        use_fade_in=False,
                        location=(0, 0),
                    )

            if event == "-COMBO-":
                if DEBUG:
                    print(values["-COMBO-"])
                if DEBUG:
                    print("EVENT IS: ", event)
                TOPIC = values["-COMBO-"]
                window["topic"].update(values["-COMBO-"])
                CONFIG["GUI"]["topic"] = values["-COMBO-"]
                save_config(config_path, CONFIG)

            if event == "-MODEL-":
                model = values["-MODEL-"]
                CONFIG["OpenAI"]["model"] = model
                save_config(config_path, CONFIG)

            if event == "-PREVIEW-":
                if values["-PREVIEW-"]:
                    window["-PREVIEW-ML-"].update(visible=values["-PREVIEW-"])
                else:
                    print(values["-PREVIEW-"])

                    window["-PREVIEW-ML-"].update(visible=values["-PREVIEW-"])

            elif event == "OK" or event == "-RETURN-":
                if event == "-ESCAPE-":
                    window.refresh()
                    break
                model = values["-MODEL-"]
                CONFIG["OpenAI"]["model"] = model

                # window["-MODEL-"].update(values["-MODEL-"])
                include_urls = values["-URLS-"]
                mem_on_off = values["memory_on_off"]
                CONFIG["GUI"]["mem_on_off"] = str(mem_on_off)
                CONFIG["GUI"]["include_urls"] = str(include_urls)
                codemode = values["code"]
                CONFIG["GUI"]["codemode"] = str(codemode)
                save_config(config_path, CONFIG)
                input_text = values["input"]
                if len(input_text) > 0:
                    # print("Input text is: ", input_text)
                    if values["topic"] != "":
                        TOPIC = values["topic"]
                if not window["-PREVIEW-ML-"].visible:
                    window.close()
                submit(
                    input_text,
                    clip,
                    img,
                    mem_on_off,
                    TOPIC,
                    codemode,
                    memory_path,
                    config_path,
                    window,
                )

                break

            elif event == "Cancel" or event == "-ESCAPE-" or None:
                if DEBUG:
                    print("Closing window")
                break

            elif event == "Reset Costs":
                reset_costs(config_path)
                window["total"].update(str(0))  # Update the total costs display to 0
                display_notification(
                    "Total Costs Reset",
                    "The total request costs have been reset to 0.",
                    img_success,  # Assuming img_success is a valid image reference for successful operation
                    5000,
                    use_fade_in=False,
                    location=(0, 0),
                )

        try:
            if window["-PREVIEW-ML-"].visible:
                while True:
                    event, values = window.read()
                    if (
                        event == sg.WIN_CLOSED
                        or event == "-ESCAPE-"
                        or event == "Cancel"
                        or event == "-RETURN-"
                    ):
                        break
        except Exception as e:
            window.close()
            display_notification(
                "Error!",
                "An exception occurred: " + str(e),
                img_error,
                5000,
                use_fade_in=False,
                location=(0, 0),
            )

        window.close()
        return

    except Exception as e:
        if (
            e is not None
            and "NoneType" not in str(type(e))
            and "subscriptable" not in str(e)
        ):
            print(e)
            print(traceback.format_exc())
            display_notification(
                "Error!",
                "An exception occurred: " + str(e),
                img_error,
                5000,
                use_fade_in=False,
                location=(0, 0),
            )
        else:
            window.close()
            return


def code_mode(reply):  # This function is used to format the reply in code mode
    if "```" not in reply:  # This if statement is used to format the reply in code mode
        return reply

    reply = reply.split("\n")
    in_code_block = False
    new_reply = []
    for line in reply:  # This loop is used to format the reply in code mode
        if (
            line.startswith("```")
            or line.startswith("<code>")
            or line.startswith("</code>")
        ):  # This if statement is used to format the reply in code mode
            in_code_block = (
                not in_code_block
            )  # This if statement is used to format the reply in code mode
            continue
        if in_code_block:  # This if statement is used to format the reply in code mode
            new_reply.append(
                line
            )  # This if statement is used to format the reply in code mode
    return "\n".join(new_reply)


def submit(
    input_text, clip, img, mem_on_off, topic, codemode, memory_path, config_path, window
):
    if not window["-PREVIEW-ML-"].visible:
        window.close()
    mem = ""
    reply = ""
    new = False
    if img:
        if os.path.exists("/tmp/copycat.jpg"):
            os.remove("/tmp/copycat.jpg")
        clip.save("/tmp/copycat.jpg")

        clip = image_to_base64("/tmp/copycat.jpg")
        clip = caption_image(clip)
    if isLink(clip.strip()) and include_urls:
        url = extracturl(clip.strip())
        if not isTwitterLink(url):
            try:
                link_summary = get_page_from_text(user, clip.strip())
                clip = clip + "\n" + "URL: " + url + "\n" + link_summary
            except Exception as e:
                print(e)
                pass
        else:
            try:
                link_summary = search_twitter(url) + get_page_from_text(
                    user, clip.strip()
                )
                clip = clip + "\n" + "URL: " + url + "\n" + link_summary
            except Exception as e:
                print(e)
                pass
    # helpmewrite(cookies, headers, user, prompt, content, pagetitle):

    if DEBUG:
        print("Mem", mem_on_off)

    if DEBUG:
        print("MODEL", model)
    cost_manager = CostManager(config_path, memory_path)

    try:
        print("topic is: ", topic)
        print("input_text is: ", input_text)
        print("clip is: ", clip)
        print("model is: ", model)
        print("mem_on_off is: ", mem_on_off)
        print("max_tokens is: ", max_tokens)
        print("temperature is: ", temperature)
        print("include_urls is: ", include_urls)
        print("codemode is: ", codemode)
        print("memory_path is: ", memory_path)
        print("config_path is: ", config_path)

        reply = cost_manager.process_request(
            topic,
            f"{input_text}\n\n{clip}",
            model,
            use_memory=mem_on_off,
            tokens=max_tokens,
            temperature=float(temperature),
        )
    except openai.error.AuthenticationError as error:
        print(error)
        display_notification(
            "Go To Settings-Preferences->OpenAI API Key",
            "An exception occurred: " + str(error),
            img_error,
            5000,
            use_fade_in=False,
            location=(0, 0),
        )
        CONFIG = load_config(config_path)
        return

    except APIError as error:
        print(error)
        display_notification(
            "Error!",
            "An exception occurred: " + str(error),
            img_error,
            5000,
            use_fade_in=False,
            location=(0, 0),
        )
        return

    if codemode:
        reply["response"] = code_mode(reply["response"])
        if DEBUG:
            print("Code mode", reply["response"])
    if DEBUG:
        print("Reply", reply["response"])
    if reply["response"].strip():  # If the reply is not empty
        if window["-PREVIEW-ML-"].visible:
            window["-PREVIEW-ML-"].update(reply["response"])
        pyperclip.determine_clipboard()  # Determine the clipboard
        pyperclip.copy(reply["response"])  # Copy the reply to the clipboard

        if not pyperclip.paste():  # If the clipboard is empty
            pyperclip.determine_clipboard()
            pyperclip.copy(reply["response"])  #
        display_notification(  # Display a notification
            "Copied to Clipboard!",  # Title
            "Your AI Request Has Completed Successfully!",  # Message
            img_success,  # Image
            5000,  # Duration
            use_fade_in=False,  # Fade in
            location=(0, 0),  # Location
            keep_on_top=True,  # Keep on top
        )

        return

    else:  # If the reply is empty
        display_notification(  # Display a notification
            "Error!",  # Title
            "Your AI Request Has Failed!",  # Message
            img_error,  # Image
            5000,  # Duration
            use_fade_in=False,  # Fade in
            location=(0, 0),  # Location
            keep_on_top=True,  # Keep on top
        )
        if not pyperclip.paste():  # If the clipboard is empty
            pyperclip.determine_clipboard()
            pyperclip.copy("")  # Copy the input text to the clipboard
        return  # Return the input text


def main(PROMPT, SKIP, prompt_user):
    while True:
        try:
            pyperclip.determine_clipboard()
            clip = pyperclip.waitForNewPaste()
            if clip == "" or clip == None and not PROMPT and not SKIP:  # IMAGE CHECK
                image = ImageGrab.grabclipboard()
                if isinstance(image, Image.Image):  # If the clipboard contains an image
                    clip = image
                    prompt_user(clip, img=True)
                    PROMPT = False
                    SKIP = False
            elif clip != "" and clip != None and not PROMPT and not SKIP:
                prompt_user(clip.strip())
                PROMPT = False
                SKIP = False
            elif SKIP:
                pyperclip.determine_clipboard()
                pyperclip.copy(clip.strip())
                SKIP = False
                PROMPT = False
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    PROMPT = False
    SKIP = False
    try:
        if DEBUG:
            print("****Main was called*****")
        main(PROMPT, SKIP, prompt_user)
    except Exception as e:
        print(e)
        display_notification(
            "Error!",
            "An exception occurred: " + str(e),
            img_error,
            5000,
            use_fade_in=False,
            location=(0, 0),
        )
