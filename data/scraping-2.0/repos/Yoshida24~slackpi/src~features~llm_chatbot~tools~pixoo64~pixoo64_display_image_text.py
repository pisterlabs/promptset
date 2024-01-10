from .pixoo64 import display
from openai.types.chat import ChatCompletionToolParam


def pixoo64_display_image_text_impl(
    image_url: str | None = None,
    text: str | None = None,
    text_color: str = "#FFFFFF",
    text_pos: str = "bottom",
    **kwargs
):
    display(image_url=image_url, text=text, text_color=text_color, text_pos=text_pos)
    return {"message": "success", "file": None}


pixoo64_display_image_text_tool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "pixoo64_display_image_text",
        "description": "Display image in URL to Pixoo, which is electronic billboard.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "Image URL. e.g. https://komori541milk.web.fc2.com/dot/4Shinnoh/474n.png",
                },
                "text": {
                    "type": "string",
                    "description": "Text to display on electronic billboard. e.g. hello world!",
                },
                "text_color": {
                    "type": "string",
                    "description": "Text color to display on electronic billboard. e.g. #FFFFFF",
                },
                "text_pos": {
                    "type": "string",
                    "description": "Text posision to display on electronic billboard. e.g. bottom",
                    "enum": ["top", "middle", "bottom"],
                },
            },
            "required": [],
        },
    },
}
