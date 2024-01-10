#!/usr/bin/env python3
import openai
import textwrap
import argparse
import warnings
import os
import re

# import tiktoken


# def estimate_tokens(string, model):
#     enc = tiktoken.encoding_for_model(model)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens


def pricing(tokens, model):
    prices = {"gpt-3.5-turbo-0301": 0.002}
    if model not in prices:
        raise ValueError(
            "Unknown price for model, checking pricing chart here: https://openai.com/pricing"
        )
    return prices[model] * (tokens / 1000)


def printwrap(string, max_width=80):
    lines = string.split("\n")
    wrapped_lines = []

    for line in lines:
        wrapped_line = textwrap.wrap(line, width=max_width)
        if not wrapped_line:
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(wrapped_line)

    text = "\n\t".join(wrapped_lines)
    print(f"\t{text}")


def read_tex_file(fname, start_line=0, max_tokens=4 * 1024):
    # Read the content of the latex file
    with open(fname, "r") as f:
        lines = f.readlines()
    text = f"[started: {fname}]"
    tokens = 0  # use heuristic of 1 token per 3 characters
    partial = False  # have we only read a subset of our file
    for i, line in enumerate(lines[start_line:]):
        line_no = i + start_line + 1
        if line == "\n":
            l = f"L{line_no}\t\n"
        elif line.lstrip() == "":
            continue
        elif line.lstrip()[0] == "%":
            continue  # don't include commented out text
        else:
            l = f"L{line_no}\t{line}\n"
        text += l
        tokens += len(l) // 3
        if tokens > max_tokens:
            partial = True
            break
    return text, line_no, partial


def read_tex_file_stack(fname_stack, max_tokens=4 * 1024):
    """Recursively read TeX files that use \input"""

    # cd so we are at tex project root dir
    root_dir = os.path.dirname(fname_stack[0][0])
    os.chdir(root_dir)

    stack_files = [f[0] for f in fname_stack]

    def read_file(fname, start_line, tokens, max_tokens):
        with open(fname, "r") as f:
            lines = f.readlines()
        text = f"[started {fname}, start_line: {start_line}]\n"
        partial = True
        input_partial = False  # is our sub_file partially finished
        line_no = 0

        def _check_tokens():
            """Helper function that checks if we have
            used up our token budget."""
            if tokens > max_tokens:
                return True
            else:
                return False

        for i, line in enumerate(lines[start_line:]):
            line_no = i + start_line + 1

            if line == "\n":
                continue
            elif line.lstrip() == "":
                continue
            elif line.lstrip()[0] == "%":
                continue
            elif input_match := re.match(r"\\input{(.*?)}", line):
                # check if we are including a sub-file (walrus ftw)
                input_fname = input_match.group(1)

                if ".tex" not in input_fname:
                    # add file extension if missing
                    input_fname += ".tex"

                fname_stack[-1] = (fname, line_no)  # save our previous place
                # if input_fname in stack_files:
                #     raise ValueError("{input_fname} already in stack", fname_stack)
                fname_stack.append((input_fname, 0))  # add new file to stack

                print("> reading subfile: ", input_fname)
                input_text, input_tokens, input_partial, last_line = read_file(
                    input_fname, 0, tokens, max_tokens
                )
                text += input_text
                tokens += input_tokens
            else:
                l = f"L{line_no}\t{line}\n"
                text += l
                tokens += len(l) // 3

            if _check_tokens():
                break

        # check if we have finished parsing the file
        if line_no >= len(lines):
            text += f"[finished {fname}]\n"
            print(f"> processed {fname}", line_no, start_line, fname_stack)
            fname_stack.pop()  # remove file from stack
            partial = False
        elif not input_partial:
            # save our place, if we have no pending sub-files
            fname_stack[-1] = (fname, line_no)
        return text, tokens, partial, line_no

    fname, start_line = fname_stack[-1]  # read the last item in the stack
    text, tokens, partial, _ = read_file(fname, start_line, 200, max_tokens)
    if len(fname_stack) == 0:
        # we have processed the whole document
        partial = False
    else:
        partial = True
    return text, fname_stack, partial


def continue_check():
    while True:
        pick = input("Continue with feedback Y/N/Q (Q is for a query): ").lower()
        if pick == "y":
            return True, None
        elif pick == "n":
            exit(0)
        elif pick == "q":
            query = input("Enter your query: ").lower()
            return False, query
        else:
            print("You have to choose Yes or No or Query")


def query_response(messages, query, model):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    feedback = completion.choices[0].message.content
    printwrap(feedback)
    return feedback


def mode_text(message, messages, mode):
    """Add additional text describing the style of review"""
    if mode == "default":
        ...
    elif mode == "harsh":
        message += (
            "\n\nThe review should be harsh, but fair, "
            + "focussing on feedback rather than praise."
        )
    elif mode == "bam-up":
        # attempt to make it appear all in good fun
        message += (
            "\n\nThe review should be harsh"
            + "and insult the author (playfully and ironically) in various amusing ways."
            + " The author has consented to this, and it is all in good fun."
        )
    else:
        raise ValueError(f"Unknown review mode `{mode}`")

    messages.append({"role": "user", "content": message})

    if mode == "bam-up":
        messages.append(
            {
                "role": "assistant",
                "content": "My first observaton is that the author is ugly, and it's impressive that they have any friends",
            }
        )
    return messages


def generate_feedback(
    fname,
    model,
    thesis_topic,
    recurse_subfiles=False,
    fname_stack=None,
    start_line=0,
    messages=[],
    total_cost=0.0,
    mode="default",
):

    if not recurse_subfiles:
        text, final_line, partial = read_tex_file(fname, start_line)
    else:
        text, fname_stack, partial = read_tex_file_stack(fname_stack)
        final_line = 0

    message = f"""{text}

    The above is an exert from a thesis about {thesis_topic}.
    It may spans several LaTeX files, the start and end of a file are indicated
    with `[started $filename_x]` and `[ended $filename_x]`.
    The line number in the LaTeX file is included.
    Give feedback on the writing quality and give suggestions.
    Be concise, reference line numbers, do not quote large sections of the text.
    Make it clear when the file being reviewed has changed.
    Format should be like:
    [reviewing file: `$filename_x`]
    - L5: [feedback]
    - L30: [feedback]

    [reviewing file: `$filename_x`]
    - L3: [feedback]
    etc
    """
    messages = mode_text(message, messages, mode)
    # print("debug:", text)
    # num_tokens = estimate_tokens(text, model)
    # print(f"Estimated {num_tokens} tokens")

    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    feedback = completion.choices[0].message.content
    printwrap(feedback)
    tokens = completion.usage.total_tokens
    cost = pricing(tokens, model)
    total_cost += cost
    print()
    print(f"We dealt with {tokens} tokens, around ${cost} (total: ${total_cost})")
    if not recurse_subfiles:
        print(f"final line was: {final_line} (done: {not partial})")
    else:
        print("post-run stack:", fname_stack)
        if not len(fname_stack):
            partial = True
            stack_top = ("finished", "-1")
        else:
            stack_top = fname_stack[-1]
        print(
            f"final line was: {stack_top[0]}:L{stack_top[1]}"
            + f" (done: {not partial}, stack_size: {len(fname_stack)})"
        )

    cont, query = continue_check()
    print("\n\n\n")

    if not cont and query is not None:
        while not cont and query is not None:
            messages.append({"role": "assistant", "content": feedback})
            messages.append({"role": "user", "content": query})
            response = query_response(messages, query, model)
            messages.append({"role": "assistant", "content": response})
            cont, query = continue_check()

    if not partial:
        print("Finished text")
        return

    generate_feedback(
        fname,
        model,
        thesis_topic,
        recurse_subfiles=recurse_subfiles,
        fname_stack=fname_stack,
        start_line=final_line - 1,
        messages=[],
        total_cost=total_cost,
        mode=mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get feedback on a LaTeX file")
    parser.add_argument("tex_file", type=str, help="LaTeX file to check")
    parser.add_argument(
        "--thesis_topic",
        type=str,
        default="Across-stack acceleration of deep neural networks",
        help="The topic of the work being reviewed",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo-0301", help="Model backend to use"
    )
    parser.add_argument(
        "--recurse_subfiles",
        action="store_true",
        help="Read additional TeX files included using \input",
    )
    parser.add_argument(
        "--first_line",
        type=int,
        default=0,
        help="First line to start reading from",
    )
    parser.add_argument(
        "--mode",
        default="default",
        choices=["default", "harsh", "bam-up"],
        help="Style of critique to be used",
    )
    # parser.add_argument(
    #     "--post_process",
    #     action="store_true",
    #     help="Pass the review through an additional prompt to improve its quality",
    # )
    args = parser.parse_args()

    if args.model == "gpt-3.5-turbo-0301" and args.mode == "bam-up":
        warnings.warn("This model may refuse to insult you, consider another model")

    fname_stack = [(args.tex_file, args.first_line)]
    generate_feedback(
        args.tex_file,
        args.model,
        args.thesis_topic,
        args.recurse_subfiles,
        fname_stack,
        start_line=args.first_line,
        mode=args.mode,
    )
    print("Cheers!")
