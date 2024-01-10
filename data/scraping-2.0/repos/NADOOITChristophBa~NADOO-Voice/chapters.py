import json
import re

import openai


def split_into_chunks(text, chunk_size=400):
    """
    Splits the book text into manageable chunks, trying to break at sentence endings.
    'chunk_size' is in characters, adjust based on testing.
    """
    chunks = []
    chunk_count = 0
    while text:
        # Take the first 'chunk_size' characters from the book text
        chunk = text[:chunk_size]

        # Ensure the chunk ends on a complete sentence where possible
        last_end = max(chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"))
        if last_end != -1 and len(chunk) - last_end < 200:
            # Adjust chunk to end at the last complete sentence
            chunk = chunk[: last_end + 1]
            # Adjust the remaining book text starting after the last complete sentence
            text = text[last_end + 1 :]
        else:
            # If no sentence ending is found, or it's too close to the end of the chunk, proceed as usual
            text = text[chunk_size:]

        chunks.append(chunk)
        chunk_count += 1

        # Print each chunk with spacing
        # print(f"Chunk {chunk_count}:\n{chunk}\n\n---\n")

    return chunks


def gpt_prompt_for_chapter_analysis(chunk, last_chapter_title):
    """
    Analyzes a text chunk to identify chapters using GPT-4, with a fallback to GPT-3.5 if necessary.
    Returns the last identified chapter if no new chapters are found, along with the text provided in the response.

    :param chunk: Text chunk to be analyzed.
    :param last_chapter_title: Title of the last identified chapter to continue from.
    :return: A list of chapters found in the chunk, or the last chapter if no new chapters are found.
    """

    from openai import (
        BadRequestError,
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
        RateLimitError,
        InternalServerError,
        APIConnectionError,
        APITimeoutError,
    )

    # Example JSON structure showing potential multiple chapters
    example_json = {
        "chapters": [
            {
                "chapter_title": "Chapter 1",
                "chapter_content": "Full found Content of Chapter 1...",
            },
            {
                "chapter_title": "Chapter 2",
                "chapter_content": "Full found Content of Chapter 2...",
            },
        ]
    }

    # Detailed prompt construction for GPT models
    prompt = (
        f"You are an helpfull AI assistant. You are helping to find the structure of a book inside a text."
        f"You are given a chunk of text. This text needs to be analysed."
        f"A chunk can contain a a chapter title but does not need to start with it."
        f"If the text does not start with a new chapter title use this title ->'{last_chapter_title}'<- for the text until you find a new chapter. "
        f"Chapter Titles usually are written in CAPITAL LETTERS and formed as a question."
        f"They also usually take a whole line."
        f"Be carful not to include any other text in the chapter title and also that in the text the chapter titles are somethimes mentioned. DO NOT include those mentions in the chapter title."
        f"Examine the text for any new chapter, and return their titles and full content. It is absolutly crucial that you return the full content of the chapters."
        f"No not change any of the text simply copy and past it."
        f"Be carfull not to add any styling to the text like /n or /t"
        f"Here is the text chunk for analysis: {chunk}."
        f"Again If no new chapters are found, simply use this ->'{last_chapter_title}'<- for the rest of the found chapter content. "
        f"Your response should be in a JSON format similar to this example: {json.dumps(example_json)}"
        f"You can do this. Give this your best shot. Take time to think. "
    )

    client = openai.OpenAI()  # Ensure the OpenAI client is set up with an API key

    attempts = 0
    max_attempts = 2
    models = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]

    while attempts < max_attempts + 1:
        model = models[attempts % len(models)]
        # print(f"Sending the following detailed prompt to {model}:")
        # print(prompt)

        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Please respond with a detailed analysis in JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        response_content = response.choices[0].message.content
        # print(f"Received response from {model}:")
        # print(response_content)

        try:
            response_data = json.loads(response_content)

            return response_data  # Correct response with new chapters

        except BadRequestError:
            print("Bad request to OpenAI. Please check the request format.")
        except AuthenticationError:
            print("Authentication failed. Please check your OpenAI API key.")
        except PermissionDeniedError:
            print("Permission denied. Please check your access rights.")
        except NotFoundError:
            print("Requested resource not found.")
        except RateLimitError:
            print("Rate limit exceeded. Please try again later.")
        except (InternalServerError, APIConnectionError, APITimeoutError) as e:
            print(f"A server or connection error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        attempts += 1

    print("Failed to get a valid response after multiple attempts.")
    return []  # Return an empty list only if all attempts fail


def word_list(text):
    # Split text into words, considering punctuation as separate entities
    return re.findall(r"\b\w+\b|\S", text.lower())


# This function was used for the extraction of chapters for the BGB book only.
def get_chapters_for_bgb_text(text):
    chapters = []
    lines = text.split("\n")
    chapter_counter = 1
    current_title = ""
    current_content = []

    for line in lines:
        # Check for Buch, Abschnitt, Titel, and start a new chapter
        if re.match(r"(Buch \d+|Abschnitt \d+|Titel \d+)", line):
            # Save previous chapter if it exists
            if current_title:
                chapters.append(
                    {
                        "unique_id": f"Chapter{chapter_counter}",
                        "chapter_title": current_title,
                        "chapter_content": " ".join(current_content).strip(),
                    }
                )
                chapter_counter += 1

            current_title = line.strip()
            current_content = []

        # Check for '§' and start a new chapter
        elif re.match(r"§ \d+", line):
            # Save previous chapter if it exists
            if current_title:
                chapters.append(
                    {
                        "unique_id": f"Chapter{chapter_counter}",
                        "chapter_title": current_title,
                        "chapter_content": " ".join(current_content).strip(),
                    }
                )
                chapter_counter += 1

            current_title = line.strip()
            current_content = []

        else:
            current_content.append(line.strip())

    # Add the last chapter
    if current_title:
        chapters.append(
            {
                "unique_id": f"Chapter{chapter_counter}",
                "chapter_title": current_title,
                "chapter_content": " ".join(current_content).strip(),
            }
        )

    return chapters


def get_chapters_for_text(text, book_title="Untitled"):
    print("Processing entire book...")

    chunks = split_into_chunks(text)
    all_chapters = []
    last_chapter_title = ""  # Initialize with an empty string

    for chunk_index, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_index + 1}: {chunk}")
        response = gpt_prompt_for_chapter_analysis(chunk, last_chapter_title)
        chapters = response.get("chapters", [])

        combined_chapter_words = []

        for chapter in chapters:
            print(f"Found chapter: {chapter.get('chapter_title')}")
            print(f"Chapter content: {chapter.get('chapter_content')}")

            title = chapter.get("chapter_title", "Untitled")
            content = chapter.get("chapter_content", "")
            last_chapter_title = title
            combined_chapter_words.extend(word_list(title + " " + content))

            chapter_found = False
            for chapter_dict in all_chapters:
                if title == chapter_dict.get("chapter_title"):
                    chapter_found = True
                    chapter_dict["chapter_content"] += " " + content
                    break
            if not chapter_found:
                all_chapters.append(
                    {"chapter_title": title, "chapter_content": content}
                )

        chunk_words = word_list(chunk)
        missing_words = [
            word for word in chunk_words if word not in combined_chapter_words
        ]

        if missing_words:
            print(f"Missing words in chunk {chunk_index + 1}: {missing_words}")

    return all_chapters
