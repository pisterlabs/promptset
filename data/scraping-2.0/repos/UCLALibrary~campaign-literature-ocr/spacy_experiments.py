import argparse
import os
from pathlib import Path
import spacy
import openai


def main():
    args = parse_args()
    ocr_files = args.ocr_files
    # Use the specified pipeline, otherwise try all
    if args.pipeline:
        pipelines = [args.pipeline]
    else:
        pipelines = get_spacy_pipelines()

    run_spacy(pipelines, root_dir="txt", file_spec=ocr_files, run_ai=args.run_ai)


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Experimental script to compare spaCy pipelines"
    )
    parser.add_argument(
        "-f", "--ocr_files", help="OCR text file(s) to process", required=True
    )
    parser.add_argument(
        "-p",
        "--pipeline",
        help="spaCy pipeline to use; uses all if not provided",
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "-a", "--run_ai", help="Run text through ChatGPT", action="store_true"
    )

    return parser.parse_args()


def get_spacy_pipelines() -> list:
    """Return list of spaCy pipelines installed."""
    pipelines = [pipeline for pipeline in spacy.info().get("pipelines")]
    return sorted(pipelines, key=lambda x: _get_pipeline_sort_order(x))


def _get_pipeline_sort_order(pipeline: str) -> int:
    """Custom sort function to allow sorting pipelines by size/complexity, just because..."""
    # Pipeline names look like en_core_web_sm, en_core_web_trf etc.
    # Get the term after the final _
    pipeline_size = pipeline.split("_")[-1]
    sort_orders = {"sm": 0, "md": 1, "lg": 2, "trf": 3}
    # If there are any unexpected/new pipelines, put them last.
    sort_order = sort_orders.get(pipeline_size, 9999)
    return sort_order


def run_spacy(
    pipelines: list,
    root_dir: str = "txt",
    file_spec: str = None,
    run_ai: bool = False,
):
    """Run spaCy with the given pipelines against each file_spec in root_dir.
    Optionally, also runs openai chatgpt against the same text.
    """
    p = Path(root_dir)
    for ocr_file in sorted(p.glob(file_spec)):
        print(f"\n{ocr_file}")
        with open(ocr_file) as f:
            ocr_text = f.read().replace("\n", " ")

            for pipeline in pipelines:
                nlp = spacy.load(pipeline)
                doc = nlp(ocr_text)
                names = {ent.text for ent in doc.ents if ent.label_ in ["PERSON"]}
                print(f"\t{pipeline}\t{names}")

            # For comparison, chatgpt data
            if run_ai:
                run_chatgpt(ocr_file, ocr_text)


def run_chatgpt(ocr_file: str, ocr_text: str):
    """Run openai chatgpt against provided text, using the prompt below."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # Basic prompt asking for "people", which probably could be improved.
    prompt = (
        "Identify the people mentioned in the following text, and provide their "
        f"full names. "
        f"For context, the original document title was {ocr_file}.\n"
        f"Document:\n{ocr_text}"
    )

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    response = chat_completion.choices[0].message.content
    print("Chat GPT:")
    for line in response.splitlines():
        print(f"\t{line}")


if __name__ == "__main__":
    main()
