import openai
import textwrap
from rich import print
from rich.console import Console
from rich.text import Text

openai.api_key = "EMPTY"
openai.api_base = "http://127.0.0.1:8080/v1"
console = Console()

MAX_WIDTH = 80  # Set maximum characters per line

def chat_prompt(user_input: str) -> str:
    return (
        'You\'re an AI assistant, respond politely and honestly to the USER.\n'
        'Only write one AI response.\n'
        f'USER: {user_input}\n'
        'AI: '
    )

def wrap_text(text, width=MAX_WIDTH, color=None):
    """Wrap text to the specified width."""
    wrapped_text = textwrap.fill(text, width)
    return f"[{color}]{wrapped_text}[/{color}]"

while True:
    # Use rich to print the User's prompt in blue
    console.print("[blue]USER: [/blue]", end="")
    user_input = input()
    
    # Exit condition if user types 'exit'
    if user_input.lower() == 'exit':
        break
    
    completion = openai.Completion.create(
        model="../aila-llama2-13b-hf",
        prompt=chat_prompt(user_input),
        max_tokens=512,
    )
    
    # Use rich to make the AI's response green and wrap it

    output_text = user_input + ' ' + completion['choices'][0]['text'].strip()

    ai_response = wrap_text(output_text.replace(user_input, '').strip(), color="green")
    print(f"[red]AILA:[/red] {ai_response}")

print("[yellow]Goodbye![/yellow]")