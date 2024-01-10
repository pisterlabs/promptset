from datetime import datetime
from openai.types.beta.threads import ThreadMessage
from rich import box
from rich.console import Console
from rich.panel import Panel


def pprint_message(message: ThreadMessage):
    """
    Pretty-prints a single message using the rich library, highlighting the
    speaker's role, the message text, any available images, and the message
    timestamp in a panel format.

    Args:
    message (dict): A message object as described in the API documentation.
    """
    console = Console()
    role_colors = {
        "user": "green",
        "assistant": "blue",
    }

    color = role_colors.get(message.role, "red")
    timestamp = datetime.fromtimestamp(message.created_at).strftime("%l:%M:%S %p")

    content = ""
    for item in message.content:
        if item.type == "text":
            content += item.text.value + "\n\n"

    # Create the panel for the message
    panel = Panel(
        content.strip(),
        title=f"[bold]{message.role.capitalize()}[/]",
        subtitle=f"[italic]{timestamp}[/]",
        title_align="left",
        subtitle_align="right",
        border_style=color,
        box=box.ROUNDED,
        # highlight=True,
        width=100,  # Fixed width for all panels
        expand=True,  # Panels always expand to the width of the console
        padding=(1, 2),
    )

    # Printing the panel
    console.print(panel)


def pprint_messages(messages: list[ThreadMessage]):
    for message in messages:
        pprint_message(message)
