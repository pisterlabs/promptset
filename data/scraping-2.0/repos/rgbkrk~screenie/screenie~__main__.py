import time

import click
from openai import OpenAI

from .audio import play_audio
from .imaging import analyze_image, prompts, take_picture, take_screenshot

client = OpenAI()


@click.command()
@click.option("--prompt", default="attenborough", help="Choice of default prompt", type=click.Choice(prompts.keys()))  # noqa
@click.option("--voice", help="Choice of voice")
@click.option(
    "--voice-provider",
    type=click.Choice(["openai", "elevenlabs"]),
    default="openai",
    help="Choice of voice provider",
)
@click.option(
    "--screenshot",
    "wants_screenshot",
    is_flag=True,
    default=False,
    help="Whether to take a screenshot",
)
@click.option(
    "--picture",
    "wants_picture",
    is_flag=True,
    default=False,
    help="Whether to take a picture",  # noqa
)
def main(prompt, voice, voice_provider, wants_screenshot, wants_picture):
    script = []

    print("Configuration: ")
    print(f"  Prompt: {prompt}")
    print(f"  Voice: {voice}")
    print(f"  Voice provider: {voice_provider}")
    print(f"  Mode: {'screenshot' if wants_screenshot else 'picture'}")

    print("Ready in 3...", end="", flush=True)
    time.sleep(1)
    print("2...", end="", flush=True)
    time.sleep(1)
    print("1...", end="", flush=True)
    time.sleep(1)
    print("Go!")

    # TODO: `play()` a starting sound

    while True:
        if wants_picture and wants_screenshot:
            raise ValueError("Can't take both a screenshot and a picture")

        # analyze screen
        print("👀 Watching...")

        if wants_picture:
            base64_image = take_picture()
        else:  # wants_screenshot is the default
            base64_image = take_screenshot()

        time.sleep(0.1)
        # Write the image to the terminal using iTerm2 inline images protocol
        print("\033]1337;File=;inline=1:" + base64_image + "\a")

        time.sleep(0.1)

        analysis = analyze_image(
            base64_image,
            script=script,
            prompt=prompt,
        )

        print("🎙️ Narrator says:")
        print(analysis)

        play_audio(analysis, provider=voice_provider, voice=voice)

        script = script + [{"role": "assistant", "content": analysis}]

        time.sleep(1)


# Run the main function
if __name__ == "__main__":
    main()
