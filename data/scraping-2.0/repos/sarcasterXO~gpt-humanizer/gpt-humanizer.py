import openai
import datetime
import pynecone as pc
from pynecone import Base

openai.api_key = 'ENTER YOUR API KEY'

class Message(Base):
    original_text: str
    text: str
    created_at: str


class State(pc.State):
    text: str = ""
    messages: list[Message] = []

    @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Humanized text will appear here."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "assistant",
                 "content": f"When it comes to writing content, two factors are crucial, \"perplexity\" and \"burstiness\". Perplexity measures the complexity of text. Separately, burstiness compares the variations of sentences. Humans tend to write with greater burstiness, for example, with some longer or complex sentences alongside shorter ones. AI sentences tend to be more uniform. Therefore, when writing the following content, I am going to ask you to create, I need it to have a good amount of perplexity and burstiness. Using the concepts written previously, rewrite this text with a high degree of perplexity and burstiness: {self.text}"}
            ]
        )

        return response['choices'][0]['message']['content']

    def post(self):
        self.messages = [
                            Message(
                                original_text=self.text,
                                text=self.output,
                                created_at=datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")
                            )
                        ] + self.messages


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("GPT-Humanizer", font_size="2rem"),
        pc.text(
            "Humanize GPT-Text to avoid AI Detection!",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.original_text),
            text_box(message.text),
            pc.box(
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def output():
    return pc.box(
        pc.box(
            smallcaps(
                "Output",
                color="#aeaeaf",
                background_color="white",
                padding_x="0.1rem",
            ),
            position="absolute",
            top="-0.5rem",
        ),
        pc.text(State.output),
        padding="1rem",
        border="1px solid #eaeaef",
        margin_top="1rem",
        border_radius="8px",
        position="relative",
    )



def index():
    """The main view."""
    return pc.container(
        header(),
        pc.input(
            placeholder="Text to humanize",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef",
        ),
        output(),
        pc.button("Humanize", on_click=State.post, margin_top="1rem"),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left",
        ),
        padding="2rem",
        max_width="600px",
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="GPT-Humanizer")
app.compile()
