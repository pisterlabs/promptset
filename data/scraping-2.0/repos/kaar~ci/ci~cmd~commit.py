from ci import git
from ci.llm import openai


def create_new_commit(history=0):
    """
    Create a new Git commit.

    Args:
        history (int): Number of latest commits to consider for generating commit message.

    Raises:
        ValueError: If the input diff is empty or git commands fail.
    """
    input_diff = git.cached_diff()
    if not input_diff:
        raise ValueError("No changes to commit.")

    history_messages = _get_history(history) if history > 0 else []
    commit_msg = _ask_for_commit_msg(input_diff.text, history=history_messages)
    if not commit_msg:
        raise ValueError("Commit message cannot be empty.")

    git.create_commit(commit_msg)


def amend_commit():
    """
    Amend the latest Git commit.

    Args:
        history (int): Number of latest commits to consider for generating commit message.

    Raises:
        ValueError: If the input diff is empty or git commands fail.
    """
    input_diff = git.cached_diff()
    if not input_diff:
        raise ValueError("No changes to commit.")

    history_messages = _get_history(1)

    commit_msg = _ask_for_commit_msg(input_diff.text, history=history_messages)

    git.amend_commit(commit_msg)


def _ask_for_commit_msg(
    input_diff: str,
    history: list,
    model: str = openai.Models.DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    """This function takes a git diff as input and returns a git commit message"""

    COMMIT_INSTRUCTION = """
You will receive a git diff and respond with a git commit message.
Provide a clear and concise commit message that summarizes the changes made in this diff.
Separate subject from body with a blank line.
Limit the subject line to 50 characters.
Capitalize the subject line.
Do not end the subject line with a period.
Use the imperative mood in the subject line.
Wrap the body at 72 characters.
Use the body to explain what and why vs. how.
"""
    history = history or []

    system_message = openai.SystemMessage(COMMIT_INSTRUCTION)
    input_message = openai.UserMessage(input_diff)

    response = openai.chat_completion(
        request=openai.ChatRequest(
            model=model,
            messages=[
                system_message,
                *history,
                input_message,
            ],
            temperature=temperature,
        )
    )
    commit_msg = response.choices[0].message.content

    return commit_msg


def _get_history(history: int) -> list[openai.Message]:
    latest_commits = git.latest_commits(history)
    history_messages: list[openai.Message] = []
    for commit in latest_commits:
        history_messages.append(openai.UserMessage(commit.message))
        history_messages.append(openai.UserMessage(commit.diff.text))

    return history_messages
