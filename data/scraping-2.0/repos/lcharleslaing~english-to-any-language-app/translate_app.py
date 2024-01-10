import os
import json
from dotenv import load_dotenv
import openai
import streamlit as st

# Load the .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def translate(text, target_language):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that translates English to various languages.",
            },
            {
                "role": "user",
                "content": f"Translate the following English text to {target_language}: {text}",
            },
        ],
    )
    return response["choices"][0]["message"]["content"]  # type: ignore


def load_settings():
    try:
        with open("settings.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "default_language": "Tagalog-Filipino",
            "history": [],
            "openai_api_key": "",
        }


def save_settings(settings):
    with open("settings.json", "w") as f:
        json.dump(settings, f)


def main():
    st.title("English to Multiple Language Translator")

    # Load settings
    settings = load_settings()

    openai.api_key = settings.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
    if not openai.api_key:
        openai.api_key = st.text_input("Please enter your OpenAI API Key")
        settings["openai_api_key"] = openai.api_key
        save_settings(settings)

    languages = [
        "Afrikaans",
        "Albanian",
        "Amharic",
        "Arabic",
        "Armenian",
        "Basque",
        "Bengali",
        "Byelorussian",
        "Burmese",
        "Bulgarian",
        "Catalan",
        "Czech",
        "Chinese",
        "Croatian",
        "Danish",
        "Dari",
        "Dzongkha",
        "Dutch",
        "English",
        "Esperanto",
        "Estonian",
        "Faroese",
        "Farsi",
        "Finnish",
        "French",
        "Gaelic",
        "Galician",
        "German",
        "Greek",
        "Hebrew",
        "Hindi",
        "Hungarian",
        "Icelandic",
        "Indonesian",
        "Inuktitut (Eskimo)",
        "Italian",
        "Japanese",
        "Khmer",
        "Korean",
        "Kurdish",
        "Laotian",
        "Latvian",
        "Lappish",
        "Lithuanian",
        "Macedonian",
        "Malay",
        "Maltese",
        "Nepali",
        "Norwegian",
        "Pashto",
        "Polish",
        "Portuguese",
        "Romanian",
        "Russian",
        "Scots",
        "Serbian",
        "Slovak",
        "Slovenian",
        "Somali",
        "Spanish",
        "Swedish",
        "Swahili",
        "Tagalog-Filipino",
        "Tajik",
        "Tamil",
        "Thai",
        "Tibetan",
        "Tigrinya",
        "Tongan",
        "Turkish",
        "Turkmen",
        "Ucrainian",
        "Urdu",
        "Uzbek",
    ]

    default_language = settings["default_language"]

    language = st.selectbox(
        "Choose Language",
        languages,
        index=languages.index(default_language),
        key="language",
    )

    if st.button("Set Default Language"):
        settings["default_language"] = language
        save_settings(settings)
        st.success(f"Default language set to {language}")

    with st.form(key="translation_form"):
        text = st.text_area("Enter your text:", key="input_text")
        submit_button = st.form_submit_button(label="Translate")

        if submit_button:
            translation = translate(text, language)
            st.text_area("Translation:", value=translation, key="translation")

            # Append to history
            settings["history"].append(
                {
                    "text": text,
                    "translation": translation,
                    "language": language,
                }
            )

            # Save history
            save_settings(settings)

    clear_button = st.button("Clear Input")
    if clear_button:
        st.experimental_rerun()

    # Display history
    if settings["history"]:
        st.subheader("Translation History")
        for item in reversed(
            settings["history"]
        ):  # Display the latest translation at the top
            st.markdown(f"**Text**: {item['text']}")
            st.markdown(f"**Translation**: {item['translation']}")
            st.markdown(f"**Language**: {item['language']}")
            st.write("---")


if __name__ == "__main__":
    main()
