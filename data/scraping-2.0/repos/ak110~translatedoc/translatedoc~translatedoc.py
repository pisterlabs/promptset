#!/usr/bin/env python3
"""translatedoc - ドキュメントを翻訳するツール。"""
import argparse
import os
import pathlib
import sys
import typing

import openai
import tqdm

if typing.TYPE_CHECKING:
    from unstructured.documents.elements import Element


def main():
    """メイン関数。"""
    # timmのimport時にSegmentation Faultが起きることがあるようなのでとりあえず暫定対策
    # https://github.com/invoke-ai/InvokeAI/issues/4041
    os.environ["PYTORCH_JIT"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        "-o",
        default=pathlib.Path("."),
        type=pathlib.Path,
        help="output directory (default: .)",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="overwrite existing files"
    )
    parser.add_argument(
        "--language",
        "-l",
        default="Japanese",
        help="target language name (default: Japanese)",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (default: OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--api-base",
        "-b",
        default=os.environ.get("OPENAI_API_URL"),
        help="OpenAI API base URL (default: OPENAI_API_URL environment variable)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=os.environ.get("TRANSLATEDOC_MODEL", "gpt-3.5-turbo-1106"),
        help="model (default: gpt-3.5-turbo-1106)",
    )

    parser.add_argument(
        "--strategy",
        "-s",
        choices=["auto", "fast", "ocr_only", "hi_res"],
        default=os.environ.get("TRANSLATEDOC_STRATEGY", "hi_res"),
        help="document partitioning strategy (default: hi_res)",
        # hi_resはtesseractやdetectron2を使うので重いけど精度が高いのでデフォルトに
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=int(os.environ.get("TRANSLATEDOC_CHUNK_MAX_CHARS", "2000")),
        help="document chunk size (default: 2000)",
    )
    parser.add_argument("input_files", nargs="+", help="input files/URLs")
    args = parser.parse_args()

    openai_client = openai.OpenAI(api_key=args.api_key, base_url=args.api_base)

    exit_code = 0
    for input_file in tqdm.tqdm(args.input_files, desc="Input files/URLs"):
        input_path = pathlib.Path(input_file)
        try:
            # ドキュメントの読み込み・パース
            tqdm.tqdm.write(f"Loading {input_file}...")
            chunks = _load_document(input_file, args)
            source_path = args.output_dir / input_path.with_suffix(".Source.txt").name
            if _check_overwrite(source_path, args.force):
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_text(
                    "\n\n".join(str(c).strip() for c in chunks) + "\n\n"
                )
                tqdm.tqdm.write(f"{source_path} written.")

            # 動作確認用: --language=noneで翻訳をスキップ
            if args.language.lower() == "none":
                continue

            # 翻訳
            output_path = (
                args.output_dir / input_path.with_suffix(f".{args.language}.txt").name
            )
            if _check_overwrite(output_path, args.force):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w") as file:
                    tqdm.tqdm.write(f"Translating {input_file}...")
                    for chunk in tqdm.tqdm(chunks, desc="Chunks"):
                        output_chunk = _translate(str(chunk), args, openai_client)
                        file.write(output_chunk.strip() + "\n\n")
                        file.flush()
                tqdm.tqdm.write(f"{output_path} written.")
        except Exception as e:
            print(f"Error: {e} ({input_file})", file=sys.stderr)
            exit_code = 1

    sys.exit(exit_code)


def _check_overwrite(output_path: pathlib.Path, force: bool) -> bool:
    """上書き確認。"""
    if output_path.exists() and not force:
        with tqdm.tqdm.external_write_mode():
            print(f"Output path already exists: {output_path}", file=sys.stderr)
            try:
                input_ = input("Overwrite? [y/N] ")
            except EOFError:
                input_ = ""
            if input_ != "y":
                print("Skipped.", file=sys.stderr)
                return False
    return True


def _load_document(input_file: str, args: argparse.Namespace) -> "list[Element]":
    """ドキュメントの読み込み・パース。"""
    with tqdm.tqdm.external_write_mode():
        from unstructured.chunking.title import chunk_by_title
        from unstructured.partition.auto import partition

    kwargs = (
        {"url": input_file}
        if input_file.startswith("http://") or input_file.startswith("https://")
        else {"filename": input_file}
    )
    elements = partition(**kwargs, strategy=args.strategy)
    chunks = chunk_by_title(
        elements,
        combine_text_under_n_chars=args.chunk_max_chars // 4,
        new_after_n_chars=args.chunk_max_chars // 2,
        max_characters=args.chunk_max_chars,
    )
    return chunks


def _translate(
    chunk: str, args: argparse.Namespace, openai_client: openai.OpenAI
) -> str:
    """翻訳。"""
    response = openai_client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": f"Translate the input into {args.language}."
                " Do not output anything other than the translation result."
                " Do not translate names of people, mathematical formulas,"
                " source code, URLs, etc.",
            },
            {"role": "user", "content": chunk},
        ],
        temperature=0.0,
    )
    if len(response.choices) != 1 or response.choices[0].message.content is None:
        return f"*** Unexpected response: {response.model_dump()=} ***"
    return response.choices[0].message.content


if __name__ == "__main__":
    main()
