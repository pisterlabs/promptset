import openai
from textual.app import App
from textual.scroll_view import ScrollView
from textual.reactive import Reactive
from textual.strip import Strip

class ChatApp(App):

    message = Reactive("")

    def on_mount(self):
        self.call_chat_completion()
        
    async def call_chat_completion(self):
        response = openai.ChatCompletion.create(
           model="gpt-3.5-turbo", 
           messages=[{"role": "user", "content": "Hello!"}],
           stream=True
        )
        async for choice in response:
            self.message = choice.message.content

    def render_line(self, y: int) -> Strip:
        return Strip(self.message)

ChatApp().run()
