# Simple object which connects to the ChatGPT API
import openai
from gi.repository import Gtk, Adw


class AiGUI(Gtk.Window):
    def __init__(self):
        super().__init__()
        self.set_title("ChatGPT")
        self.set_size_request(200, 300)


class AiConnector:
    def __init__(self, api_key, model):
        # TODO: This costs moolah
        openai.api_key = api_key
