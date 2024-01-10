import openai
import pygments
from prompt_toolkit import prompt
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_by_name


def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def lightgrey(text):
    return colored(211, 211, 211, text)


def main():
    print("Chat started. Press Alt-Enter or Esc follwed by Enter to send message.")
    messages = []
    while True:
        print()
        query = prompt("> ", multiline=True)
        if query == "" or query == "exit" or query == "quit":
            break
        messages.append({"role": "user", "content": query})
        # print(messages)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        messages.append({"role": "assistant", "content": completion.choices[0].message.content})
        response = completion.choices[0].message.content

        final = []
        for i, text in enumerate(response.split("```")):
            if i % 2 == 1:
                lang = text.split("\n")[0]
                try:
                    lexer = get_lexer_by_name(lang, stripall=True)
                    formatter = TerminalFormatter()
                    final.append("\n" + highlight(text[len(lang) :], lexer, formatter))
                except pygments.util.ClassNotFound:
                    final.append(lightgrey(text))
            else:
                final.append(lightgrey(text))

        print("```".join(final))


if __name__ == "__main__":
    main()
