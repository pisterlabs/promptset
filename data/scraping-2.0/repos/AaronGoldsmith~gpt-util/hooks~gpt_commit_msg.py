"""
This module is used as a Git hook to generate commit messages using OpenAI GPT-3.5-turbo.
It takes a diff as input, processes it into chunks, and generates a commit message summary.
"""

import sys
import os
import openai
import tiktoken


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


ADDED_CODE = sys.argv[1]


def main():
    """
    The main function processes the input diff, divides it into chunks,
    and generates commit messages using the OpenAI API.
    It then prints the generated commit messages.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    code_chunks = ADDED_CODE.split('diff --git')

    summaries = []
    combined_chunks = []

    # Combine code_chunks until the token limit is reached
    current_chunk = ''
    for chunk in code_chunks:
        if chunk:
            temp_chunk = current_chunk + 'diff --git' + chunk
            tokens = num_tokens_from_string(temp_chunk)

            if tokens <= 4096:
                current_chunk = temp_chunk
            else:
                combined_chunks.append(current_chunk)
                current_chunk = 'diff --git' + chunk

    # Add the last chunk to the combined_chunks list
    if current_chunk:
        combined_chunks.append(current_chunk)

    for chunk in combined_chunks:
        system_intro = "You are a commit message generator used in prepare-commit-msg.\n \
                        1. Review the provided diff\n  \
                        2. Respond with <commit message summarizing +/- changes> \n \
                        3. Return \"Commit Title\nCommit Description (max 120 characters)\" separated by a new line \
                        Constraint: Respond with the unpunctuated commit message +\
                              brief commit description of the changes.\n"

        prompt = f"{chunk}\n<<END_DIFF>>\n\n"

        if tokens > 4096:
            print(f"Warning: Skipping file '{code_chunks}' due to token\
                   limit exceeded ({tokens} tokens).", file=sys.stderr)
            continue

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=40,
            messages=[
                {"role": "system", "content": system_intro},
                {"role": "user", "content": prompt}
            ]
        )
        summary = completion.choices[0].message.content
        summaries.append(summary)

    generated_commit_message = " ".join(summaries)
    print(generated_commit_message)

    return generated_commit_message


if __name__ == "__main__":
    main()
