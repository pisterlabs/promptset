import csv
import io
from itertools import combinations
import math
import re
from openai.embeddings_utils import cosine_similarity
import os

import click

from .openai import complete_chat


class LanguageError(Exception):
    """Abstract base class for errors related to a constructed language."""

    pass


class DictionaryError(LanguageError):
    """Exception raised when there is an error with a dictionary."""

    pass


class CreateDictionaryError(DictionaryError):
    """Exception raised when a dictionary cannot be created."""

    pass


class ImproveDictionaryError(DictionaryError):
    """Exception raised when the dictionary cannot be updated."""

    pass


class NoDictionaryError(DictionaryError):
    """
    Exception raised when a dictionary is not found in a response from ChatGPT.
    """

    pass


class InvalidDictionaryError(DictionaryError):
    """Exception raised when a dictionary is invalid."""

    pass


class TranslationError(LanguageError):
    """Exception raised when there is an error with a translation."""

    pass


def _get_related_words(text, dictionary, embeddings_model):
    """Get the most related words from the dictionary."""

    click.echo(
        click.style(f"Getting the most relevant words from the dictionary...", dim=True)
    )

    # Calculate the cosine similarity between each word in the text and each
    # word in the dictionary (both the word and its English translation)
    words_in_text = text.split()
    word_embeddings = [
        embeddings_model.embed_documents([word])[0] for word in words_in_text
    ]
    related_words = []
    for word_embedding in word_embeddings:
        word_similarities = {}
        for word, translation in dictionary.items():
            word_similarity = cosine_similarity(
                word_embedding, embeddings_model.embed_documents([word])[0]
            )
            translation_similarity = cosine_similarity(
                word_embedding, embeddings_model.embed_documents([translation])[0]
            )
            similarity = max(word_similarity, translation_similarity)
            word_similarities[word] = similarity

        if not word_similarities:
            continue

        most_related_word = sorted(
            word_similarities, key=word_similarities.get, reverse=True
        )[0]

        # Skip words that aren't similar enough
        if word_similarities[most_related_word] < 0.75:
            continue

        related_words.append(most_related_word)

    return related_words


def _parse_dictionary_from_paragraph(paragraph, similarity_threshold, embeddings_model):
    reader = csv.reader(io.StringIO(paragraph))
    # Skip the header row
    next(reader)

    # Parse the words
    dictionary = {}
    for row in reader:
        # Extract the word and translation
        if len(row) != 2:
            raise NoDictionaryError(
                f"Invalid response. Expected row to have two columns: Word and Translation. Received: {row}"
            )
        word, translation = row

        # If the word starts with a number (e.g., "1. hello"), remove the number
        if "." in word:
            first, rest = word.split(".", 1)
            if first.isdigit():
                word = rest.strip()

        # Remove any trailing whitespace
        word = word.strip()
        translation = translation.strip()

        # Add the word to the dictionary
        dictionary[word] = translation

    # Ensure that the dictionary is not empty
    if not dictionary:
        raise NoDictionaryError(
            f"Invalid response. Expected a CSV document with at least one data row. Received: {paragraph}"
        )

    # Remove similar words
    dictionary = reduce_dictionary(dictionary, similarity_threshold, embeddings_model)

    return dictionary


def _parse_dictionary(text, similarity_threshold, embeddings_model):
    # Sometimes, ChatGPT returns multiple paragraphs. If this happens, use the
    # first paragraph that can be parsed as a dictionary.
    if "\n\n" in text:
        click.echo(text)
        for paragraph in text.split("\n\n"):
            try:
                dictionary = _parse_dictionary_from_paragraph(
                    paragraph, similarity_threshold, embeddings_model
                )
                break
            except NoDictionaryError:
                pass
        else:
            raise NoDictionaryError(text)

    else:
        dictionary = _parse_dictionary_from_paragraph(
            text, similarity_threshold, embeddings_model
        )

    return dictionary


def translate_text(text, language_guide, dictionary, model, embeddings_model):
    """Translate text into a constructed language."""

    # Get the most related words from the dictionary
    related_words = _get_related_words(text, dictionary, embeddings_model)

    # Translate the text
    click.echo(click.style(f"Translating text using {model}...", dim=True))
    if related_words:
        formatted_related_words = "\n".join(
            [f"- {word}: {dictionary[word]}" for word in related_words]
        )
        click.echo(
            click.style(f"Most related words:\n\n{formatted_related_words}", dim=True)
        )
        chat_completion = complete_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the text below from or into the following constructed language. Explain how you arrived at the translation. Only use words found in either the guide or the list below. Wrap the final translation with <translation> and </translation>.\n\nLanguage guide:\n\n{language_guide}\n\nPotentially-related words:\n\n{formatted_related_words}\n\nText to translate:\n\n{text}",
                }
            ],
            temperature=0,
        )
        response = chat_completion["choices"][0]["message"]["content"]

    else:
        chat_completion = complete_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the text below from or into the following constructed language. Explain how you arrived at the translation. Only use words found in the guide.\n\nNo relevant words from dictionary found. Wrap the final translation with <translation> and </translation>.\n\nLanguage guide:\n\n{language_guide}\n\nText to translate:\n\n{text}",
                }
            ],
            temperature=0,
        )
        response = chat_completion["choices"][0]["message"]["content"]

    # Parse the translation
    if (
        "<translation>" not in response.lower()
        or "</translation>" not in response.lower()
    ):
        # If the translation is not wrapped in <translation> and </translation>,
        # the response is probably an explanation of why the text cannot be
        # translated.
        raise TranslationError(response)

    # Extract the final translation
    translated_text = re.split("<translation>", response, flags=re.IGNORECASE)[1]
    translated_text = re.split("</translation>", translated_text, flags=re.IGNORECASE)[
        0
    ]
    # Remove the xml markup from the explanation
    explanation = re.sub("<translation>", "", response, flags=re.IGNORECASE)
    explanation = re.sub("</translation>", "", explanation, flags=re.IGNORECASE)

    return translated_text, explanation


def generate_english_text(model):
    click.echo(
        click.style(f"Generating random English text using {model}...", dim=True)
    )
    chat_completion = complete_chat(
        model=model,
        temperature=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        messages=[
            {
                "role": "system",
                "content": "You are a writing assistant who likes to write about different topics.",
            },
            {"role": "user", "content": "Please generate a random English sentence."},
        ],
    )

    english_text = chat_completion["choices"][0]["message"]["content"]
    return english_text


def improve_language(guide, dictionary, model, embeddings_model, text=None):
    click.echo(click.style(f"Improving language using {model}...", dim=True))

    if text is None:
        # Identify problems with the language
        chat_completion = complete_chat(
            model=model,
            temperature=0.5,
            presence_penalty=0.5,
            messages=[
                {
                    "role": "user",
                    "content": f'If the language outlined below has any flaws, contradictions or points of confusion, please identify one and provide specific, detailed, actionable steps to fix it. Otherwise, respond with "No problem found" instead of the csv document. Note that a dictionary is provided separately.\n\nLanguage guide:\n\n{guide}',
                }
            ],
        )
        revisions = chat_completion["choices"][0]["message"]["content"]

    else:
        # Attempt to translate the provided text
        try:
            _, explanation = translate_text(
                text, guide, dictionary, model, embeddings_model
            )
        except TranslationError as e:
            explanation = str(e)

        click.echo(f"Translation:\n\n{explanation}\n")

        # Identify problems with the language using the translated text as an example/reference
        chat_completion = complete_chat(
            model=model,
            temperature=0.1,
            # TODO: Replace with `presence_penalty=0.1`
            frequency_penalty=0.1,
            messages=[
                {
                    "role": "user",
                    "content": f'If the language outlined below has any flaws, contradictions or points of confusion, please identify one and provide specific, detailed, actionable steps to fix it. Otherwise, respond with "No problem found". I included a sample translation to give you more context.\n\nLanguage guide:\n\n{guide}\n\nOriginal text: {text}\n\nTranslated text: {explanation}',
                }
            ],
        )
        revisions = chat_completion["choices"][0]["message"]["content"]

    # Check if any problems were found
    if "No problem found" in revisions or "No problems found" in revisions:
        click.echo("No problems found.")
        return None

    click.echo(f"Change:\n\n{revisions}\n")

    # Rewrite the language guide
    chat_completion = complete_chat(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": f"Improve the following constructed language to address the problem described below. Your response should be a reference sheet describing the new language's rules. Assume the reader did not read the original reference sheet and that they have no prior experience using the language. Note that a dictionary is provided separately.\n\nOriginal language guide:\n\n{guide}\n\nMake these changes:\n\n{revisions}",
            }
        ],
    )
    improved_guide = chat_completion["choices"][0]["message"]["content"]

    return improved_guide


def generate_language(design_goals, model):
    """Generate a constructed language."""

    click.echo(f"Generating language using {model}...")
    chat_completion = complete_chat(
        model=model,
        temperature=0.9,
        presence_penalty=0.5,
        messages=[
            {
                "role": "user",
                "content": f"Create a constructed language with the following design goals:\n\n{design_goals}\n\nYour response should be a reference sheet including all the language's rules. Assume the reader has no prior experience using the language.",
            }
        ],
    )
    guide = chat_completion["choices"][0]["message"]["content"]
    click.echo(f"Initial draft:\n\n{guide}\n")

    return guide


def modify_language(guide, changes, model):
    """Apply specified changes to a constructed language."""

    click.echo(click.style(f"Modifying language using {model}...", dim=True))
    chat_completion = complete_chat(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": f"Make the following changes to the constructed language outlined below. Your response should be a reference sheet describing the new language's rules. Assume the reader did not read the original reference sheet and that they have no prior experience using the language. Note that a dictionary is provided separately.\n\nOriginal language guide:\n\n{guide}\n\nMake these changes:\n\n{changes}",
            }
        ],
    )
    improved_guide = chat_completion["choices"][0]["message"]["content"]

    return improved_guide


def reduce_dictionary(words, similarity_threshold, embeddings_model):
    """Remove similar words from a dictionary."""

    click.echo(click.style(f"Removing similar words using local model...", dim=True))

    # Retrieve the embeddings for each word
    translation_embeddings = {
        word: embeddings_model.embed_query(translation)
        for word, translation in words.items()
    }

    # Remove similar words
    words_to_remove = set()
    for (word_a, embedding_a), (word_b, embedding_b) in combinations(
        translation_embeddings.items(), 2
    ):
        if cosine_similarity(embedding_a, embedding_b) > similarity_threshold:
            click.echo(
                click.style(
                    f"Removing {word_b} ({words[word_b]}) because it is too similar to {word_a} ({words[word_a]}).",
                    dim=True,
                )
            )
            words_to_remove.add(word_b)

    # Remove the similar words from the dictionary
    for word in words_to_remove:
        words.pop(word)

    return words


def create_dictionary_for_text(
    guide, text, existing_dictionary, similarity_threshold, model, embeddings_model
) -> dict:
    """Generate words for a constructed language."""

    click.echo(click.style(f"Generating words using {model}...", dim=True))

    # Get related words from the existing dictionary
    related_words = _get_related_words(text, existing_dictionary, embeddings_model)

    # Format the related words as a CSV document
    mutable_formatted_related_words = io.StringIO()
    writer = csv.writer(mutable_formatted_related_words)
    writer.writerow(["Conlang", "English"])
    for word in related_words:
        writer.writerow([word, existing_dictionary[word]])
    formatted_related_words = mutable_formatted_related_words.getvalue()

    # Generate words
    user_message = None
    if len(related_words) > 0:
        click.echo(f"Related words:\n\n{formatted_related_words}\n")
        user_message = f"Create any new words required to translate the following text into the constructed language outlined below. The Conlang-to-English dictionary is lazy-generated. The words you create will be saved to this dictionary. Write each new word in its root form. Omit words that can be derived from existing words in the dictionary. Omit all proper nouns, except for common words (such as days of the week). In general, the conlang word should not resemble its English translation. Your response should be a CSV document with any new words. The document should start with the following header: Conlang,English. Any following rows should have exactly two cells and contain each new word with its translation.\n\nLanguage guide:\n\n{guide}\n\nText to translate (either from or to the conlang):\n\n{text}\n\nExisting words that could be related:\n\n{formatted_related_words}"
    else:
        user_message = f"Create all the root words required to translate the following text into the constructed language outlined below. The Conlang-to-English dictionary is lazy-generated. None of the words found in the text currently have translations in this dictionary. The words you create will be saved there. Write each word in its root form. In general, the conlang word should not resemble its English translation. YYour response should be a CSV document beginning with the following header: Conlang,English. Each row should have exactly two cells.\n\nLanguage guide:\n\n{guide}\n\nEnglish text:\n\n{text}."

    chat_completion = complete_chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
        temperature=1,
        presence_penalty=2,
    )
    response = chat_completion["choices"][0]["message"]["content"]

    # Parse the generated words
    try:
        words = _parse_dictionary(response, similarity_threshold, embeddings_model)
    except NoDictionaryError as e:
        # If no dictionary was returned, ChatGPT probably stated that no new
        # words were required to translate the text. Return an empty dictionary,
        # but print the message from ChatGPT.
        click.echo(click.style(str(e), fg="yellow"))
        return {}

    # Remove new words that already had translations in the conlang
    existing_english_words = set(
        [word.lower() for word in existing_dictionary.values()]
    )
    words_to_remove = set()
    for conlang_word, english_word in words.items():
        if english_word.lower() in existing_english_words:
            click.echo(
                click.style(
                    f"Removed {conlang_word} because it already had a translation.",
                    dim=True,
                )
            )
            words_to_remove.add(conlang_word)
    for word in words_to_remove:
        del words[word]

    # Regenerate duplicate conlang words
    while True:
        # Get the words that are already in the dictionary
        existing_conlang_words = set([word.lower() for word in existing_dictionary])

        # Get the words that are in the dictionary but have different translations
        conflicting_words = {}
        for word in words:
            if word.lower() in existing_conlang_words:
                conflicting_words[word] = existing_dictionary[word]

        # If there are no conflicting words, stop
        if len(conflicting_words) == 0:
            break

        # Remove the conflicting words from the dictionary
        for word in conflicting_words:
            del words[word]

        # Format the conflicting words as a CSV document
        mutable_formatted_conflicting_words = io.StringIO()
        writer = csv.writer(mutable_formatted_conflicting_words)
        writer.writerow(["Conlang", "English"])
        for word, translation in conflicting_words.items():
            writer.writerow([word, translation])
        formatted_conflicting_words = mutable_formatted_conflicting_words.getvalue()

        # Regenerate the conflicting words
        click.echo(
            click.style(
                f"Regenerating {len(conflicting_words)} conflicting words...",
                dim=True,
            )
        )
        chat_completion = complete_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Replace the following conlang words with completely new ones. The translations should remain exactly the same. Your response should be a CSV document with two columns: Conlang and English. Each row should have exactly two cells.\n\nLanguage guide:\n\n{guide}\n\nWords to regenerate:\n\n{formatted_conflicting_words}",
                }
            ],
            temperature=1,
            presence_penalty=2,
        )
        response = chat_completion["choices"][0]["message"]["content"]

        # Parse the regenerated words
        try:
            regenerated_words = _parse_dictionary(
                response, similarity_threshold, embeddings_model
            )
        except NoDictionaryError as e:
            # If no dictionary was returned, ChatGPT probably stated that no new
            # words were required to translate the text. Return an empty dictionary,
            # but print the message from ChatGPT.
            click.echo(click.style(str(e), fg="yellow"))
            return {}

        # Add the regenerated words to the dictionary
        for word, translation in regenerated_words.items():
            words[word] = translation

    # Print the new words
    if words:
        formatted_words = "\n".join(
            [f"- {word}: {translation}" for word, translation in words.items()]
        )
        click.echo(f"New words:\n\n{formatted_words}\n")

    return words


def improve_dictionary(
    dictionary, guide, similarity_threshold, model, embeddings_model, batch_size=25
):
    """Update the dictionary to match the guide by focusing on updating the words themselves instead of their translations."""

    click.echo(click.style(f"Improving dictionary using {model}...", dim=True))

    # Get the words to improve
    words_to_improve = list(dictionary.keys())

    # Improve the words
    for i in range(0, len(words_to_improve), batch_size):
        # Dump the batch to a csv string
        mutable_batch_string = io.StringIO()
        writer = csv.writer(mutable_batch_string)
        writer.writerow(["Conlang", "English"])
        for word in words_to_improve[i : i + batch_size]:
            if not word in dictionary:
                continue
            writer.writerow([word, dictionary[word]])
        formatted_batch = mutable_batch_string.getvalue()

        # Improve the words
        chat_completion = complete_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f'Ensure that the following words are correctly translated into the constructed language outlined below. If any of the words do not adhere to the guide below, update them as you see fit. Your response should be a CSV document with two columns: Conlang and English. Each row represents an updated word and should have exactly two cells. If the word list below is correct and complete, respond with "No problems found".\n\nLanguage guide:\n\n{guide}\n\nWords to improve:\n\n{formatted_batch}',
                }
            ],
            temperature=0,
            presence_penalty=1,
        )
        response = chat_completion["choices"][0]["message"]["content"]

        # Parse the response
        if "No problems found" in response:
            continue

        try:
            improved_words = _parse_dictionary(
                response, similarity_threshold, embeddings_model
            )
        except NoDictionaryError as e:
            raise ImproveDictionaryError from e

        # Update the words in the dictionary with the improved words
        for improved_word, improved_translation in improved_words.items():
            for existing_word in words_to_improve[i : i + batch_size]:
                if not existing_word in dictionary:
                    continue

                existing_translation = dictionary[existing_word]
                if improved_translation.lower() == existing_translation.lower():
                    click.echo(
                        click.style(
                            f"Updated {existing_word} to {improved_word}.", dim=True
                        )
                    )
                    del dictionary[existing_word]
                    dictionary[improved_word] = improved_translation

    return dictionary


def merge_dictionaries(a, b, similarity_threshold, embeddings_model):
    """Merge two vocabulary dictionaries, removing similar words."""

    click.echo(click.style(f"Merging dictionaries using local model...", dim=True))

    # Retrieve the embeddings for each translation
    a_embeddings = {
        word: embeddings_model.embed_query(translation)
        for word, translation in a.items()
    }
    b_embeddings = {
        word: embeddings_model.embed_query(translation)
        for word, translation in b.items()
    }

    # Calculate the cosine similarity between each pair of translations
    a = dict(a)
    b = dict(b)
    similarities = {}
    for a_word, a_embedding in a_embeddings.items():
        for b_word, b_embedding in b_embeddings.items():
            similarities[(a_word, b_word)] = cosine_similarity(a_embedding, b_embedding)

    # Remove words whose translations are too similar. Prefer shorter words.
    for (a_word, b_word), similarity in similarities.items():
        # Skip words that have already been removed
        if a_word not in a or b_word not in b:
            continue

        if similarity > similarity_threshold:
            if len(b_word) < len(a_word):
                click.echo(
                    click.style(
                        f"Removing {a_word} ({b[b_word]}) because it is too similar to {b_word} ({b[b_word]}).",
                        dim=True,
                    )
                )
                del a[a_word]
            else:
                click.echo(
                    click.style(
                        f"Removing {b_word} ({b[b_word]}) because it is too similar to {a_word} ({a[a_word]}).",
                        dim=True,
                    )
                )
                del b[b_word]

    # Merge the dictionaries
    merged = {**a, **b}

    return merged


def load_dictionary(dictionary_path):
    # Load the dictionary
    if os.path.exists(dictionary_path):
        with open(dictionary_path, "r") as file:
            reader = csv.reader(file)

            # Skip the header row
            next(reader)

            # Load the dictionary
            dictionary = {row[0]: row[1] for row in reader}
    else:
        dictionary = {}

    return dictionary


def save_dictionary(dictionary, dictionary_path):
    # Save the dictionary in alphabetical order
    with open(dictionary_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Word", "Translation"])
        for word in sorted(dictionary.keys()):
            writer.writerow([word, dictionary[word]])
