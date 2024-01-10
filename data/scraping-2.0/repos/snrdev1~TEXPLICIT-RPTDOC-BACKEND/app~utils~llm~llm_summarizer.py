import openai

from app.utils.llm.base import break_up_text


def summarize_text(text, max_tokens=450):
    """
    The `summarize_text` function takes in a text and breaks it up into chunks, then uses the OpenAI API
    to generate a summary for each chunk, and finally returns the combined summaries as a single string.
    
    Args:
      text: The `text` parameter is the input text that you want to summarize. It can be a long piece of
    text or multiple paragraphs.
      max_tokens: The `max_tokens` parameter specifies the maximum number of tokens that the generated
    summary should have. Tokens are chunks of text that can be as short as one character or as long as
    one word. The OpenAI API has a limit on the number of tokens that can be processed in a single API
    call. Defaults to 450
    
    Returns:
      The function `summarize_text` returns a string that represents the summarized version of the input
    text.
    """
    summaries = []

    for chunk in break_up_text(text):
        print("Summarizing...")
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=chunk,  # + "\n\nTl;dr",
            max_tokens=max_tokens,
            temperature=0,
        )
        summary = response["choices"][0]["text"].strip().lstrip(": ")
        summaries.append(summary)

    return " ".join(summaries)
