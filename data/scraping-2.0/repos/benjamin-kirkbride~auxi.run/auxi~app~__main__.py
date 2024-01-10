import logging
from pathlib import Path

import openai
from faster_whisper import WhisperModel

from compare_models import config

logger = logging.getLogger(__name__)

TOP_P = 0.1
TEMPERATURE = 1


def _get_transcription(model_name: str, input_file: Path) -> str:
    logger.info(f"initializing model '{model_name}'")
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    logger.info(f"model '{model_name}' initialized")

    segments, info = model.transcribe(str(input_file), beam_size=5)
    segment_list = list(segments)
    logger.info("transcription completed")

    logger.info(
        f"Detected language '{info.language}' with"
        f" probability {info.language_probability}",
    )

    return "".join([segment.text for segment in segment_list])


def _summarize(transcription: str) -> str:
    system_prompt = """
        You are an agent tasked with process transcripts of a single person's
        stream of consciousness.

        You use Markdown to format your output.

        You evaluate the contents of the transcription and summarize what was said
        using headings and bullet lists as appropriate.

        You organize the headings and bullets logically, and not necessarily in
        the order in which they were said.

        If appropriate you have a specific heading for action items, tasks, or goals.
    """

    return openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcription},
        ],
        top_p=TOP_P,
        temperature=TEMPERATURE,
    )


def _main():
    file = Path(__file__).parent.parent.parent / "test.m4a"
    assert file.is_file()

    transcript = _get_transcription(
        model_name="large-v2",
        input_file=file,
    )

    logger.info(f"{transcript=}")

    output_file_name_format = "summary-%s.md"

    for path in [
        Path(Path(f"scratch-{TEMPERATURE}-{TOP_P}") / (output_file_name_format % i))
        for i in range(10)
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        response = _summarize(transcription=transcript)
        summary = response["choices"][0]["message"]["content"]
        with path.open("w") as file:
            file.write(summary)


if __name__ == "__main__":
    _main()
