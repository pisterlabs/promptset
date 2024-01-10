import panflute as pf
import re
import sys

def my_filter(doc):
    bad_blocks = [pf.BlockQuote]
    filtered_doc = []
    for _, x in enumerate(doc):
        if type(x) in bad_blocks:
            continue
        markdown_output = pf.convert_text(
            x, input_format="panflute", output_format="markdown"
        )
        markdown_output_single_line = re.sub(
            r"([^\n])\n([^\n])", r"\1 \2", markdown_output
        )
        markdown_output_single_line = re.sub(r"  +", r" ", markdown_output_single_line)
        filtered_doc.append(markdown_output_single_line)
        # filtered_doc += markdown_output_single_line.split('. ')
    return filtered_doc


def split_into_chunks(paragraphs, max_chars):
    """
    Split a list of paragraphs into chunks.

    :param paragraphs: List of paragraphs (each paragraph is a string).
    :param max_chars: Maximum number of characters allowed in a chunk.
    :return: List of chunks (each chunk is a list of paragraphs).
    """

    chunks = []
    current_chunk = []

    current_length = 0
    for para in paragraphs:
        para_length = len(para)

        # Check if the paragraph itself is too long
        if para_length > max_chars:
            raise ValueError(
                f"Paragraph exceeds the maximum character limit: {para_length} characters"
            )

        # Check if adding this paragraph would exceed the max length
        if current_length + para_length <= max_chars:
            current_chunk.append(para)
            current_length += para_length
        else:
            # Start a new chunk
            chunks.append(current_chunk)
            current_chunk = [para]
            current_length = para_length

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


from openai import OpenAI
import openai


def gpt_proofread(text, model="gpt-4-1106-preview"):
    prompt = """
You are a proofreader. The user provides a piece of R-markdown, and you will proofread it. Do not say anything else.
Do not change the style, change a word/phrase to a "more formal" one, "fluff up the prose", make it "more serious", make it "more academic", or change the tone. Only fix grammar and awkward flow. If you change the flow or grammar, you must STILL preserve the word choice and style. Do not use a more formal word just because you have fixed the grammar or flow.
For example, "use" -> "utilize" is bad, "gave" -> "provided" is bad... That's fluffing up the prose with formality. We don't need formality. We need clarity.
You MUST reply in this format:

a: <original sentence>
b: <rewritten sentence>

a: <original sentence>
b: <rewritten sentence>

...

Reply only rewritten sentences. Every sentence MUST be on one line ONLY, that is, both <original sentence> and <rewritten sentence> MUST contain no newline character.
Every rewritten sentence MUST differ from its original sentence.
Do not use American-style quotation. Use logical quotation. Do not use single quotation marks.
Do not use en-dash or em-dashes -- use double or triple hyphens. Do not use unicode ellipsis -- use three dots.
""".strip()
    example_user_1 = """
We use the convention putting derivative on the rows – for convenience… This convention simplifies a lot of equations, and completely avoids transposing any matrix.

In the next section, using the "pebble construction," they studied "Gamba perceptrons." They stated "MLPs are essentially Gamba perceptrons."
    """.strip()
    example_assistant_1 = """
a: We use the convention putting derivative on the rows – for convenience…
b: We use the convention of putting the derivatives on the rows -- for convenience...

a: In the next section using the "pebble construction," they studied "Gamba perceptrons."
b: In the next section, using the "pebble construction", they studied "Gamba perceptrons".

a: They stated "MLPs are essentially Gamba perceptrons."
b: They stated "MLPs are essentially Gamba perceptrons.".
    """
    example_user_2 = "That is, for any $X \subset R$, we have $\psi(X)=\psi(g(X))$."
    example_assistant_2 = ""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": example_user_1},
            {"role": "assistant", "content": example_assistant_1},
            {"role": "user", "content": example_user_2},
            {"role": "assistant", "content": example_assistant_2},
            {"role": "user", "content": text},
        ],
    )
    return response


from fuzzysearch import find_near_matches
from warnings import warn


# Despite trying my best, GPT4 still sometimes returns a line that is not in the original text.
# and sometimes it returns a "modified" line that is literally the same as the original line.
# This function tries to fix the first problem.
# The second problem is not fixed here, though it is easier to fix by filtering the proofread.txt file.
def process_response(response, original_text="", max_l_dist=10):
    text = response.choices[0].message.content
    # Check that the input sequence satisfies a certain format
    prefix_list = [line.strip()[:2] for line in text.split("\n") if line.strip() != ""]
    prefix_string = "".join(prefix_list)
    if not re.fullmatch(r"((\?|a):b:(c:)?)*", prefix_string):
        raise ValueError(f"Illegal response sequence:\n{prefix_string}")

    fixed_text_tuples = []
    for line in text.split("\n"):
        if line.strip() == "":
            continue
        if ":" not in line:
            raise ValueError(
                f"Illegal response:\n{stripped_line}\n{'-'*80}\nFound in text:\n{text}"
            )
        prefix = line.split(":")[0]
        if prefix not in ["a", "b", "c"]:
            raise ValueError(
                f"Illegal response:\n{stripped_line}\n{'-'*80}\nFound in text:\n{text}"
            )

        stripped_line = line[len(prefix) + 1 :].strip()
        if prefix == "a" and original_text and stripped_line not in original_text:
            warn(
                f"The proofreader returned a line of inexact match:\n{stripped_line}",
                UserWarning,
            )
            fuzzy_search_result = find_near_matches(
                stripped_line, original_text, max_l_dist=max_l_dist
            )
            if fuzzy_search_result == []:
                prefix = "?"
                warn(
                    f"The proofreader returned a line of unknown origin:\n{stripped_line}",
                    UserWarning,
                )
            else:
                stripped_line = fuzzy_search_result[0].matched
        fixed_text_tuples.append((prefix, stripped_line))

    fixed_text = ""
    for prefix, stripped_line in fixed_text_tuples:
        if prefix in "?a":
            fixed_text += "\n"
        fixed_text += f"{prefix}: {stripped_line}\n"
    fixed_text = f"\n\n{fixed_text.strip()}"
    return fixed_text


def get_proofread_files(input_file, proofread_file, max_chars=4_000):
    with open(input_file, "r", encoding="utf8") as file:
        input_markdown = file.read()
    doc = pf.convert_text(input_markdown)
    try:
        chunks = split_into_chunks(my_filter(doc), max_chars=max_chars)
        print(f"{len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            input_string = "\n\n".join(chunk).strip()
            if input_string == "":
                continue
            response = gpt_proofread(input_string)
            response_text = process_response(response)
            length = len(response_text.split("\n\n"))
            print(f"{length} responses received")
            with open(proofread_file, "a", encoding="utf8") as file:
                file.write(response_text)
    except ValueError as e:
        raise e

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: script.py <input_file> <output_file> <max_chars>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_chars = int(sys.argv[3])

    get_proofread_files(input_file, output_file, max_chars=max_chars)
