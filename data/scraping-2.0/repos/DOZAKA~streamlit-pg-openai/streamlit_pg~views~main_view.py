import streamlit
from .content_view import ContentView
from .content_view.custom_api_view import CustomApiView
from .content_view.image_generations_view import ImageGenerationsView
from .content_view.image_variations_view import ImageVariationsView
from .content_view.moderation_view import ModerationContentView
from .content_view.papgo_api_view import PapagoApiView
from streamlit_option_menu import option_menu
from streamlit_pg.common import ExtendedEnum
from streamlit_pg.modules.openai_manager.openai_manager import OpenAIManager


class ModelType(ExtendedEnum):
    CUSTOM_API = "CUSTOM_API"
    IMAGE_GENERATIONS = "IMAGE_GENERATIONS"
    IMAGE_VARIATIONS = "IMAGE_VARIATIONS"
    TEXT_MODERATION = "TEXT_MODERATION"
    PAPAGO_API = "PAPAGO_API"

class MainView:
    def __init__(self, st: streamlit, oai_manager: OpenAIManager):
        self.st = st
        self.oai_manager: OpenAIManager = oai_manager
        self.selected_type: ModelType = ModelType.IMAGE_GENERATIONS
        self.content_view: ContentView = None

    def view(self) -> None:
        selected_type: ModelType = ModelType(self.side_view())

        if selected_type == ModelType.CUSTOM_API:
            self.content_view = CustomApiView(self.st)
        elif selected_type == ModelType.IMAGE_GENERATIONS:
            self.content_view = ImageGenerationsView(self.st, self.oai_manager)
        elif selected_type == ModelType.IMAGE_VARIATIONS:
            self.content_view = ImageVariationsView(self.st, self.oai_manager)
        elif selected_type == ModelType.TEXT_MODERATION:
            self.content_view = ModerationContentView(self.st, self.oai_manager)
        elif selected_type == ModelType.PAPAGO_API:
            self.content_view = PapagoApiView(self.st)

        self.content_view.view()

    def side_view(self) -> str:
        self.st.sidebar.title("Streamlit Playground for OpenAI")

        with self.st.sidebar:
            return option_menu(
                menu_title="Test Models",
                options=ModelType.values(),
                icons=["image", "image-alt", "chat-dots"]
            )
