"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import pynecone as pc
import openai

openai.api_key = "" # removed key for security purpose


class State(pc.State):
    """The app state."""

    prompt = ""
    image_url = ""
    image_processing = False
    image_made = False

    def process_image(self):
        """Set the image processing flag to true and indicate that the image has not been made yet."""
        self.image_made = False
        self.image_processing = True

    def get_image(self):
        """Get the image from the prompt."""
        try:
            # response = openai.Image.create(prompt=self.prompt, n=1, size="1024x1024")
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Convert the following text to emojis: \"{self.prompt}\"",
                temperature=0.8,
                max_tokens=60,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            print(response["choices"][0]["text"])
            self.image_url = response["choices"][0]["text"]
            # Set the image processing flag to false and indicate that the image has been made.
            self.image_processing = False
            self.image_made = True
        except Exception as e:
            self.image_processing = False
            return pc.window_alert(f"Error with OpenAI Execution. {e}")


def index():
    return pc.center(
        pc.vstack(
            pc.heading("Emoji Story", font_size="1.5em"),
            pc.input(placeholder="Enter a prompt..", on_blur=State.set_prompt),
            pc.button(
                "Generate Emojis",
                on_click=[State.process_image, State.get_image],
                width="100%",
            ),
            pc.divider(),
            pc.cond(
                State.image_processing,
                pc.circular_progress(is_indeterminate=True),
                pc.cond(
                    State.image_made,
                    pc.text(State.image_url)
                ),
            ),
            bg="white",
            padding="2em",
            shadow="lg",
            border_radius="lg",
        ),
        width="100%",
        height="100vh",
        background="radial-gradient(circle at 22% 11%,rgba(62, 180, 137,.20),hsla(0,0%,100%,0) 19%),radial-gradient(circle at 82% 25%,rgba(33,150,243,.18),hsla(0,0%,100%,0) 35%),radial-gradient(circle at 25% 61%,rgba(250, 128, 114, .28),hsla(0,0%,100%,0) 55%)",
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Emoji_Story")
app.compile()
