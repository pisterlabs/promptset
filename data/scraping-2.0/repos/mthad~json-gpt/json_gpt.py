import openai, os
from lark import Lark, UnexpectedToken, UnexpectedCharacters

openai.api_key = os.getenv("OPENAI_KEY")

# JSON CFG grammar
# We set "start: object" to always get an object out.
# We could also do "start: value" if we'll accept any
# json value as output from the model.
json_grammar = r"""
    ?start: object

    ?value: object
          | array
          | string
          | SIGNED_NUMBER      -> number
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null

    array  : "[" [value ("," value)*] "]"
    object : "{" [pair ("," pair)*] "}"
    pair   : string ":" value

    string : ESCAPED_STRING

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS

    %ignore WS
"""


def generate_content(
    prompt,
    model="text-davinci-003",
    # Initialize the CFG parser with LALR and standard lexer
    # These are the fastest parsers, and suffice for the json grammar.
    parser = Lark(json_grammar, parser="lalr", lexer="basic"),
    # The lexer doens't like incomplete strings. This leaves us with the issue of
    # not knowing if a bad generation has occoured, or we just are in a long string.
    # The current hack is to give up if we have seen too many characters without
    # the lexer being able to progress.
    max_string_length = 50,
    max_tokens=193,
    temperature=0.7,
    # GPT models are pretty good at outputting valid json, so for demo purposes
    # we can add some errors manually to the output.
    demo_mode = False,
    verbose = True,
):
    """
    Generate content from the model and validate each chunk using a CFG parser.
    """

    # If we have a generation like
    #    "tags": ["non-fiction", "science"],}
    # The parser error happens at "}" and not at ",". This means we'd potentially
    # force the model too keep generating items to the object, instead of going
    # back and getting rid of the ",".
    # Our solution is to restart a little bit earlier than our current position.
    # Each time an error happens, we'll increase the amount we step back.
    trim_amount = 5

    current_text = ""
    while True:
        buffer = ""
        parser_instance = parser.parse_interactive()
        try:
            for token in parser.lex(current_text):
                parser_instance.feed_token(token)
        except UnexpectedCharacters as e:
            # If the current_text doesn't completely parse, just parse as much as we can,
            # and leave the rest in the buffer.
            buffer = current_text[e.pos_in_stream :]
            current_text = current_text[: e.pos_in_stream]
        except UnexpectedToken as e:
            print("Current text already flawed", e)
            raise

        response = openai.Completion.create(
            model=model,
            prompt=prompt + current_text + buffer,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for event in response:
            event_text = event["choices"][0]["text"]

            # Introduce an error, since the model doesn't do it on it's own.
            if demo_mode and event_text.endswith("]"):
                event_text += ","
                demo_mode = False

            if not event_text:
                return current_text

            if verbose:
                print(event_text, end="", flush=True)

            buffer += event_text
            consumed_length = 0
            lexer_instance = parser.lex(buffer)
            try:
                for token in lexer_instance:
                    parser_instance.feed_token(token)
                    consumed_length = token.end_pos
            except UnexpectedToken as e:
                keep = max(0, len(current_text) - trim_amount)
                removed = len(current_text) - keep
                current_text = current_text[:keep]
                # Increase the trim amount of next time. This prevents us from getting stuck in a loop.
                trim_amount += 1
                if verbose:
                    print(
                        f"\n\nError: {e}\nTrimming away {removed} chars. Now ...{repr(current_text[-10:])}\n"
                    )
                # Break out of the loop to restart the Completion with current content
                break
            except UnexpectedCharacters as e:
                # If we have unexpected characters, we simply stop parsing and go back
                # to increasing the size of the buffer, then go back to parsing later.
                pass

            # Transfer consumed characters from buffer to current_text.
            current_text += buffer[:consumed_length]
            buffer = buffer[consumed_length:]

            # Check that we are not getting too many UnexpectedCharacters errors.
            if len(buffer) > max_string_length:
                break

    raise "Should never get here"


def main():
    prompt = """
    Title: "The Great Gatsby"
    Author: F. Scott Fitzgerald
    Publication Year: 1925
    Tags: fiction, classic

    Title: "Murder on the Orient Express"
    Author: Agatha Christie
    Publication Year: 1934
    Tags: fiction, mystery

    Title: "A Brief History of Time"
    Author: Stephen Hawking
    Publication Year: 1988
    Tags: non-fiction, science

    Let's write this list of books using the following JSON schema:

    {
      books: [
        {
          title: string,
          author: string,
          publicationYear: integer,
          tags: [string],
        }
      ]
    }

    json ="""

    print(prompt, end="")
    output = generate_content(prompt, demo_mode=True, verbose=True)
    print("\nFinal output: json =", output)


if __name__ == "__main__":
    main()
