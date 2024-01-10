from typing import Union

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM

from music_generator.generate_section import generate_section
from music_generator.generate_section_effects import generate_section_effects
from music_generator.music_generator_types.markup_types import (
    MusicalMarkup,
)
from music_generator.music_generator_types.base_song_types import Song


def generate_song(
    musical_markup: MusicalMarkup, llm: Union[BaseChatModel, BaseLLM], key: int = 0
) -> Song:
    song = Song()
    sections = musical_markup.sections

    # Note: This relies on the fact that the sections are ordered. Each section might reference a previous section.
    for section in sections.keys():
        generated_section = generate_section(
            markup_section=sections[section], prev_gens=song, llm=llm
        )
        generated_effects = generate_section_effects(
            markup_section=sections[section],
            number_bars=len(generated_section.bars),
            llm=llm,
        )
        generated_section.apply_effects(generated_effects)

        song.append_section(generated_section)

    return song


if __name__ == "__main__":
    from music_generator.generate_markup import generate_markup
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.chat_models import ChatOpenAI
    from music_generator.music_generator_types.base_song_types import Config

    from dotenv import dotenv_values

    config = Config(**dotenv_values())  # type: ignore

    musical_markup = generate_markup(
        song_description="""Create an outline for a house music track""".strip(),
        llm=ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model="gpt-4",
            temperature=0.70,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        ),
    )

    song = generate_song(
        llm=ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model="gpt-4",
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        ),
        musical_markup=musical_markup,
    )
    print(song)
