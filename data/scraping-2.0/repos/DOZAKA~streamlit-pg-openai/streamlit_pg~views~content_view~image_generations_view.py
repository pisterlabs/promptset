import streamlit
from . import ContentView
from streamlit_pg.modules.openai_manager.openai_manager import OpenAIManager


class ImageGenerationsView(ContentView):
    def __init__(self, st: streamlit, oai_manager: OpenAIManager):
        super().__init__(st, oai_manager)
        self.openai_doc: str = "https://platform.openai.com/docs/guides/images/generations"

    def view(self) -> None:
        self.st.write("OpenAI API Introduction [link](%s)" % self.openai_doc)
        self.st.header("[Request]")
        content: str = self.st.text_area("Type a Text to generate image")

        if not self.st.button('Send Request'):
            return

        self.st.header("[Response]")

        img_url: str = self.oai_manager.text_to_image_url(content)

        if not img_url:
            return

        self.st.image(img_url)
