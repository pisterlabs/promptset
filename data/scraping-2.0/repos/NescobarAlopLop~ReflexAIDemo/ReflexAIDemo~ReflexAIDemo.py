import reflex as rx
from loguru import logger
import openai
import os

# Get the value of an environment variable named 'MY_VARIABLE'
value_from_env_variable = os.getenv("OPEN_AI_API")
value_from_env_variable = os.getenv("OPENAI_API_KEY")
openai.api_key = value_from_env_variable


class State(rx.State):
    prompt = ""
    image_url = ""
    processing = False
    complete = False

    def get_image(self):
        if self.prompt == "":
            logger.warning("Prompt is empty.")
            return rx.window_alert("Prompt Empty")

        logger.info("Starting image processing.")
        self.processing, self.complete = True, False
        yield

        try:
            response = openai.Image.create(prompt=self.prompt, n=1, size="512x512")
            self.image_url = response["data"][0]["url"]
            logger.success("Image successfully created.")
        except openai.error.OpenAIError as e:
            logger.error(f"An error occurred while interacting with the OpenAI API: {e}")
            self.image_url = None  # You can handle the error as needed here
        finally:
            self.processing, self.complete = False, True
            logger.info("Image processing complete.")


def index():
    return rx.center(
        rx.vstack(
            rx.heading("DALL·E"),
            rx.input(placeholder="Enter a prompt", on_blur=State.set_prompt),
            rx.button(
                "Generate Image",
                on_click=State.get_image,
                is_loading=State.processing,
                width="100%",
            ),
            rx.cond(
                State.complete,
                rx.image(
                    src=State.image_url,
                    height="25em",
                    width="25em",
                ),
            ),
            padding="2em",
            shadow="lg",
            border_radius="lg",
        ),
        width="100%",
        height="100vh",
    )


# Add state and page to the app.
app = rx.App()
app.add_page(index, title="reflex:DALL·E")
app.compile()
