from ai_curses import handle_messages as hm
from ai_curses import openai
import requests
import shutil
import ai_curses.command_prompt as command_prompt


def blank_line():
    return " \n"


def main_help(app, args, command, messages):
    return f"""
{blank_line()}
:help - Show this help.
:history - Show current history stats.
:settings - Show the current settings of this session.
:prompt - Show the current system prompt (super command).
:forget - Remove the oldest message from the message list,
          if your chat has gotten too big.
:image [PROMPT] - Create an image with a prompt. You must provide a prompt,
and there must be an output directory set in the config or with the -o flag.
{blank_line()}
Type 'quit' or 'exit' to exit the program.
{blank_line()}
"""


def prompt(app, args, command, messages):
    prompt = args.super.replace('"""', '').replace('\n', '')
    return f"""
{blank_line()}
Prompt:
{prompt}
{blank_line()}
"""


def image(app, args, command, messages):
    image_prompt = command.split(":image")[1].strip()
    if len(image_prompt) < 1 or not args.output_dir:
        return f"""
{blank_line()}
You must provide a prompt, and an output directory must be set.
{blank_line()}
example: :image a pretty swan
{blank_line()}
your prompt: :image {image_prompt}
output directory: {args.output_dir}
"""
    else:
        app.panels["prompt"].clear()
        app.screen.refresh()
        command_prompt.title(app, processing=True)
        response = openai.get_image(image_prompt, timeout=args.timeout)
        if response.status_code == 200:
            image_url = response.json().get('data')[0]['url']
            res = requests.get(image_url, stream=True)
            dest = f"{args.output_dir}/{image_prompt}.png"
            if res.status_code == 200:
                with open(dest, 'wb') as f:
                    shutil.copyfileobj(res.raw, f)
                    f.close()
            else:
                return f"""
Image successfully generated, but could not be downloaded:

Image URL: {image_url}
"""
            with open(args.output_md, 'a', encoding='utf-8') as f:
                f.write("Human> Image Prompt: {} \n\n".format(image_prompt))
                f.write("AI> \n\n ![[{}]] \n\n".format(f"{image_prompt}.png"))
                f.close()
            return f"Image saved to: {args.output_dir}/{image_prompt}.png"
        else:
            return f"""Image generation failed: \n{response.text}"""


def settings(app, args, command, messages):
    if not args.output_md:
        output_md = "none output file set"
    else:
        output_md = f"\"{args.output_md}\""
    if not args.output_json:
        output_json = "no output json file set"
    else:
        output_json = f"\"{args.output_json}\""
    return f"""
{blank_line()}
Settings:
{blank_line()}
Output folder - {args.output_dir}
Output Markdown File - {output_md}
Output JSON file - {output_json}
Timeout - {args.timeout} seconds
{blank_line()}
"""


def history(app, args, command, messages):
    role_counts = {"system": 0, "assistant": 0, "user": 0, "unknown": 0}
    for msg in messages:
        role_counts[msg.get('role', 'unknown')] += 1
    return f"""
{blank_line()}
History:
{blank_line()}
# of messages - {len(messages)}
   - system: {role_counts.get('system')}
   - assistant: {role_counts.get('assistant')}
   - user: {role_counts.get('user')}
   - unknown: {role_counts.get('unknown')}
{blank_line()}
"""


def help_menu():
    return {
        "help": main_help,
        "history": history,
        "settings": settings,
        "prompt": prompt,
        "image": image
    }


def handler(app, args, command, messages):
    meta_command = command.split(" ")[0][1:]
    if help_menu().get(meta_command, None):
        hm.add_to_chat_output(
            app, help_menu().get(meta_command, lambda: 'Invalid')(
                app, args, command, messages
            ),
            "green_on_black"
        )
    else:
        hm.add_to_chat_output(
            app,
            f"{blank_line()}You tried a meta-command called "
            f"\"{meta_command}\".\n"
            f"{blank_line()}Unfortunately, I don't know that "
            f"one!\n{blank_line()}",
            "green_on_black"
        )
