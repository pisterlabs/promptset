import openai


def clean_text(text):
    """
    Cleans the extracted text from PDF using OpenAI GPT-4.

    Parameters:
        text (str): The extracted text.

    Returns:
        str: The cleaned text.
    """

    system_prompt = """
        You are a specialized text fixer for invoices. Your task is to:
        - Correct any spelling or grammatical errors.
        - Reorder any text that may be in the wrong sequence.
        - Separate words that are combined and combine words that are split.
        - Extract and organize only the essential structured data such as:
            - Customer details (Name)
            - Item lists (Description, Quantity, Price)
            - Totals (Subtotal, Taxes, Final Total)
        - Remove any redundant or unnecessary information.
        - Return the cleaned and structured text in a way that's easy to read and analyze.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{text}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    cleaned_text = response['choices'][0]['message']['content'].strip()

    return cleaned_text

