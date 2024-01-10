import json
import sys
import click
import openai
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.rule import Rule
# to enable command line editing using GNU readline.
import readline


@click.command()
@click.argument("system_prompt", type=str, required=False)
@click.option('--no-stream', 'stream', is_flag=True, default=True,
              help="if no stream, token usage will be shown.")
@click.option("-o", "--output", 'output_path', type=str,
              help="save chat history(json format) to which file")
@click.option("-i", "--input", 'input_path', type=str,
              help="load history(json format) from which input file, chat with context")
def main(system_prompt: str, stream: bool, output_path: str, input_path: str):
    console = Console()
    messages = []
    if input_path:
        with open(input_path, 'r') as f:
            messages.extend(json.load(f))
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    while True:
        # user input
        try:
            s = input("🙋Please Input: ")
            if s == '':
                console.print("multiline input, ctrl+d to finish, ctrl+c to exit ")
                s = sys.stdin.read()
                console.print()
            user_msg = {"role": "user", "content": s}
        except KeyboardInterrupt:
            break

        # model output
        try:
            with Live(Spinner(name="dots", text="connecting...", style="green"),
                      transient=True, console=console):
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages + [user_msg],
                    stream=stream,
                )
            console.print(Rule(title="ChatGPT:", align="left", style="cyan"))
            if stream:
                msg = ""
                with Live(console=console, auto_refresh=False) as live:
                    for chunk in response:
                        msg += chunk['choices'][0]['delta'].get('content', '')
                        live.update(Markdown(msg), refresh=True)
                console.print(Rule(style="cyan"), "")
            else:
                msg = response['choices'][0]['message']['content']
                usage = response['usage']
                console.print(
                    Markdown(msg),
                    Rule(title=f"token prompt:{usage['prompt_tokens']}, completion:{usage['completion_tokens']},"
                               f" total:{usage['total_tokens']}", style="cyan", align="right"), "")
        except KeyboardInterrupt:
            console.log("model response stopped by user, ctrl+c again to exit")
        except (openai.error.AuthenticationError, openai.error.PermissionError) as e:
            console.log(e)
            break
        except openai.error.RateLimitError as e:
            console.log("OpenAI API busy: ", e)
        except openai.error.OpenAIError as e:
            console.log("OpenAI Error: ", e)
        else:
            messages.append(user_msg)
            messages.append({'role': 'assistant', 'content': msg})
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(messages, f)
    console.print("\nbye")


if __name__ == '__main__':
    main()
