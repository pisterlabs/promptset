"""Produce vocabulary lists from audio files, using OpenAI API."""

from pathlib import Path
import logging
import fire
import openai
import whisper
import torch
import pydub
import tempfile
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

AVAILABLE_MODELS = ["gpt-3.5-turbo", "gpt-4"]
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_WHISPER_MODEL = "medium"

# TODO Change the prompt depending on the language.
# Could even do this with the API itself
PROMPT_START = (
    """Hello, ChatGPT: you are now a helpful language learning expert, who has perfect knowledge of French. """
    """You are given the following list of words, which were transcribed from an audio CD accompanying a French vocabulary book.\n\n"""
    """The below transcription may be imperfect, so any errors may be silently corrected."""
    """It is your job to turn the following transcription into a vocabulary list with French and English translations side by side, in a CSV format.\n\n"""
)

TIMED_EXAMPLE_INPUT_FR = (
    """
    ```
    Le nom. [3.76 - 4.96]
    Le nom de famille. [6.44 - 9.18]
    Le prénom. [11.08 - 11.82]
    S'appeler. [14.68 - 15.30]
    ```
    """
)

TIMED_EXAMPLE_OUTPUT_FR = (
    """
    ```
    "French","English","start","end"
    "le nom", "the name",3.76,4.96
    "le nom de famille","the last name, family name",6.44,9.18
    "le prénom","the first name",11.08,11.82
    "s'appeler","to be called, to be named",14.68,15.30
    ```
    """
)

PROMPT_END = (
    """For example, for the input"""
    f"{TIMED_EXAMPLE_INPUT_FR}\n"
    """would be:\n"""
    f"{TIMED_EXAMPLE_OUTPUT_FR}\n\n"
    """When a masculine and feminine nouns or adjectives follow each other, they should be merged into one entry."""
    """Similarly, if a plural noun follows its singular version, they should both appear side-by-side in the same entry."""
    """All strings in the CSV should be escaped with `"`."""
    """Capitalise all proper nouns and valid grammatical sentences/questions, but otherwise leave the entries lowercase."""
    """For the valid grammatical sentences/questions, also punctuate them correctly.\n\n"""
    """Respond only in CSV format."""
)

# Prompt for imposing sensible punctuation output from Whisper
TRANSCRIPTION_PROMPT = (
    """Le nom. Le nom de famille. Le prénom. S'appeler. Comment t'appelles-tu? Comment tu t'appelles? Monsieur. Messieurs. Madame. Mesdames. Madame Martin, née Dupont. Mademoiselle. Médemoiselle. Habiter quelque chose."""
)

# TODO Move this constant to a separate place
DEFAULT_SILENCE_THRESHOLD = -40

# Number of lines that fit into 4k context length GPT API call
# TODO Tune this number
MAX_LEN_4K = 90


def get_prompt(transcription: str) -> str:
    """Given a transcription, return a prompt for ChatGPT to produce a vocabulary list.

    Args:
        transcription: Transcription of audio file.

    Returns:
        Prompt for ChatGPT to produce a vocabulary list.
    """
    return PROMPT_START + "\n\n" + transcription + "\n\n" + PROMPT_END


def create_vocab_list(transcription: str,
                      model: str,
                      section_name: str = "") -> str:
    """Given a transcription, return a vocabulary list using OpenAI API.

    Args:
        transcription: Transcription of audio file.
        model: OpenAI model to use.
        section_name: Name of section to use in logging.

    Returns:
        Vocabulary list produced by OpenAI API.
    """
    # Increasing context length if transcription is too long
    if len(transcription.splitlines()) > MAX_LEN_4K and model == "gpt-3.5-turbo":
        # NOTE This costs more than "gpt-3.5-turbo", but cheaper than splitting into multiple calls
        model = "gpt-3.5-turbo-16k"

    prompt = get_prompt(transcription)
    logging.debug(prompt)

    # This is where the API call is made
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    logging.info("Made OpenAI `%s` call for section %s", model, section_name)
    vocab_list = response["choices"][0]["message"]["content"]
    finish_reason = response["choices"][0]["finish_reason"]
    if finish_reason != "stop":
        logging.warning("OpenAI API call for %s did not finish properly. Finish reason: %s", section_name, finish_reason)

    return vocab_list


def get_audio_list(audio_path: Path) -> list[Path]:
    """Return list of all mp3 audio files in a given path.

    Args:
        audio_path: Path to audio file or directory containing audio files.

    Returns:
        audio_paths_all: List of all audio files in a given path.

    Raises:
        ValueError: If audio_path is neither a directory nor a file.
    """
    audio_paths_all: list[Path] = []

    # Treat differently depending on whether audio_path is a directory or a file
    if audio_path.is_dir():
        audio_paths_all = sorted(audio_path.glob("*.mp3"))
    elif audio_path.is_file():
        audio_paths_all.append(audio_path)
    else:
        raise ValueError("audio_path must be a directory or a file.")
    return audio_paths_all


def get_root_name(audio_path: Path, strip_numbers: bool) -> str:
    """Return root name of audio file."""
    root_name = audio_path.stem
    if strip_numbers:
        root_name = root_name.lstrip("0123456789. ")
    return root_name


def get_audio_without_start(audio_path: Path) -> tuple[pydub.AudioSegment, float]:
    """Return audio without the first segment leading up to silence."""
    audio = pydub.AudioSegment.from_mp3(audio_path)
    first_10_seconds = audio[:10000]

    # TODO Add customisable silence threshold
    silences = pydub.silence.detect_silence(first_10_seconds, min_silence_len=600, silence_thresh=DEFAULT_SILENCE_THRESHOLD)
    start_time = silences[0][1] - 300  # 300ms before the first silence
    if len(silences) > 0:
        audio = audio[start_time:]

    return audio, (start_time / 1000)


# TODO Add the full type hint
def split_sentences(whisper_output: dict[list], offset: float = 0.0) -> list[dict[str, str | float]]:
    """Split Whisper output into sentences with timestamps.

    TODO Fix this so it's not so dependent on punctuation.

    Args:
        whisper_output: Output from Whisper API call when word-level timestamps are enabled.
        offset: Offset to add to timestamps. (Useful if audio has been trimmed at the start.)
    """
    sentences = []
    sentence_start = 0.0

    for segment in whisper_output["segments"]:
        word_list = segment["words"]

        current_sentence = []

        for word_dict in word_list:
            word = word_dict["word"]
            # Get start of segment
            if len(current_sentence) == 0:
                sentence_start = word_dict["start"]

            # Adding word to current sentence
            current_sentence.append(word)

            # If word ends with punctuation, add sentence to list and reset current sentence
            # TODO Add more robust regex matching for punctuation
            # NOTE This causes a bug when Whisper doesn't output punctuation
            if word.endswith(".") or word.endswith("?") or word.endswith("!"):
                construct_sentence = "".join(current_sentence).strip()
                sentences.append({
                    "text": construct_sentence,
                    "start": sentence_start + offset,
                    "end": word_dict["end"] + offset,
                })
                current_sentence = []
    return sentences


def format_sentences(sentences: list[dict[str, str | float]]) -> str:
    """Format sentences into a string along with timestamp."""
    lines = []
    for sentence in sentences:
        lines.append(f"{sentence['text']} [{sentence['start']:.2f} - {sentence['end']:.2f}]")
        # append line but with float rounded to 2 decimal places

    return "\n".join(lines)


def write_transcription(transcription: str, output_path: Path) -> None:
    """Write transcription to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcription)
        logging.info("Saved transcription to %s", output_path.name)


class VocabularyList:
    """Class holding functions for generating Anki deck.

    Methods:
        create_transcriptions: Given an audio file, create a transcription.
        create_vocab_lists: Given an audio file, create a vocabulary list.
    """
    def __init__(self,
                 api_key_path: Path = None,
                 api_key: str = None,
                 strip_numbers: bool = False,
                 language: str = "fr",
                 local: bool = True,
                 model: str = DEFAULT_MODEL,
                 save_raw: bool = False) -> None:
        """Initialise VocabularyList class.

        Args:
            api_key_path: Path to OpenAI API key.
            api_key: OpenAI API key.
            strip_numbers: Whether to strip leading numbers from section names.
            language: ISO code of target language.
            local: Whether to use local instance of Whisper instead of API.
            model: OpenAI model to use.
            save_raw: Whether to save raw transcription from Whisper call without timestamps.
        """
        # Set API key
        if api_key_path is not None:
            openai.api_key_path = api_key_path
        elif api_key is not None:
            openai.api_key = api_key

        self.strip_numbers = strip_numbers
        self.model = model
        if model not in ["gpt-3.5-turbo", "gpt-4"]:
            self.model = DEFAULT_MODEL
            logging.warning("Model %s not recognised. Using default model %s.", model, self.model)

        self.language = language
        self.local = local
        self.save_raw = save_raw
        if self.local:
            # TODO Add MPS support and selectable device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda" and torch.cuda.mem_get_info()[0] > 10e9:
                # Use large model if GPU has enough memory
                self.whisper_model_name = "large"
            else:
                self.whisper_model_name = DEFAULT_WHISPER_MODEL
            logging.info("Using local '%s' Whisper model on device '%s'", self.whisper_model_name, self.device)
            self.whisper_model = whisper.load_model(self.whisper_model_name, device=self.device)
            # NOTE Conditioning on previous text might be necessary for prompt to be followed
            self.whisper_kwargs = {"condition_on_previous_text": False, "word_timestamps": True, "initial_prompt": TRANSCRIPTION_PROMPT}
        else:
            self.whisper_model_name = "whisper-1"
            self.whisper_kwargs = {"model": self.whisper_model_name, "prompt": TRANSCRIPTION_PROMPT}
            self.whisper_model = openai.Audio
        self.whisper_kwargs["language"] = self.language

    def transcribe_audio(self,
                         audio_path: Path,
                         offset: float = 0.0,
                         ) -> tuple[str, str]:
        """Given an audio file, return a transcription."""
        if not self.local:
            with open(audio_path, "rb") as file:
                logging.info("Made OpenAI Whisper API call with audio_file %s", audio_path.name)
                # This is where an API call happens
                transcription = self.whisper_model.transcribe(
                                                file=file,
                                                **self.whisper_kwargs,
                                        )
                transcription_text = transcription["text"]
                return transcription_text, transcription_text
        else:
            logging.info("Made local Whisper call on %s with audio file %s", self.device, audio_path.name)

            # TODO Sort failure when using local model on "02.1 Body Parts, Organs"
            transcription = self.whisper_model.transcribe(
                str(audio_path),
                **self.whisper_kwargs,
            )

            # Keeping raw transcription in case punctuation is missing
            transcription_text = transcription["text"]
            logging.info("%s", transcription_text)

            # Split into sentences with timestamps
            sentences = split_sentences(transcription, offset=offset)
            return transcription_text, format_sentences(sentences)

    def create_transcriptions(self,
                              audio_path: Path,
                              output_path: Path,
                              no_split_start: bool = False,
                              ) -> None:
        """Given an audio file, create a transcription with timestamps.

        Args:
            audio_path: Path to audio file or directory containing audio files.
            output_path: Path to output directory.
            no_split_start: Whether to split audio file at start of each section. (Useful to remove confusing English.)
        """
        transcription_path = Path(output_path) / "transcriptions"
        transcription_path.mkdir(parents=True, exist_ok=True)
        for audio_file in tqdm(get_audio_list(Path(audio_path))):
            with tempfile.TemporaryDirectory() as temp_dir:
                start_time = 0.0
                # Create new file without start
                if not no_split_start:
                    audio_no_start, start_time = get_audio_without_start(audio_file)
                    audio_file = Path(temp_dir) / audio_file.name
                    audio_no_start.export(audio_file, format="mp3")

                section_name = get_root_name(audio_file, self.strip_numbers)

                # This is where the API call is made
                transcription_raw, transcription_timestamped = self.transcribe_audio(audio_file, offset=start_time)

                # Save transcription
                write_transcription(transcription_timestamped, transcription_path / (section_name + ".txt"))
                if self.save_raw:
                    write_transcription(transcription_raw, transcription_path / "raw" / (section_name + ".txt"))

        logging.info("Finished transcribing audio files.")

    def create_vocab_lists(self,
                           audio_path: Path,
                           output_path: Path,
                           ) -> None:
        """Given an audio file, create a vocabulary list.

        TODO Add an option to use transcription directly. Split up this function.

        Args:
            audio_path: Path to audio file or directory containing audio files.
            output_path: Path to output directory.
            language: Language of audio file.
        """
        output_path = Path(output_path)
        transcription_dict = {}
        transcription_path = output_path / "transcriptions"
        for audio_file in get_audio_list(Path(audio_path)):
            # Check if transcription exists
            section_name = get_root_name(audio_file, self.strip_numbers)
            transcription_file = transcription_path / (section_name + ".txt")
            if not transcription_file.exists():
                # Create transcription if it doesn't exist
                self.create_transcriptions(audio_file, output_path)

            # Read transcription
            transcription: str = ""
            with open(transcription_file, "r", encoding="utf-8") as f:
                transcription = f.read()

            transcription_dict[section_name] = transcription

        for section_name, transcription in transcription_dict.items():
            vocab_list = create_vocab_list(transcription, self.model, section_name=section_name)

            vocab_path = output_path / "vocab"
            vocab_path.mkdir(parents=True, exist_ok=True)
            with open(vocab_path / (section_name + ".csv"), "w", encoding="utf-8") as f:
                f.write(vocab_list)
                logging.info("Saved vocabulary list for %s", section_name)
        logging.info("Finished creating vocabulary lists.")


if __name__ == "__main__":
    fire.Fire(VocabularyList)
