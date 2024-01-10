import copy
import os
import sys

from gptwntranslator.api import openai_api
from gptwntranslator.helpers.config_helper import Config
from gptwntranslator.helpers.file_helper import write_md_as_epub
from gptwntranslator.helpers.text_helper import parse_chapters, write_novel_md
from gptwntranslator.origins.origin_factory import OriginFactory
from gptwntranslator.storage.json_storage import JsonStorage, JsonStorageException, JsonStorageFileException, JsonStorageFormatException
from gptwntranslator.translators.gpt_translator import GPTTranslatorSingleton

def setup() -> None:
    print("Initializing... ", end="")
    sys.stdout.flush()
    cf = Config()
    storage = JsonStorage()
    storage.initialize(cf.vars["persistent_file_path"])
    cf.load(cf.vars["config_file_path"])
    cf.vars["target_language"] = cf.get_language_name_for_code(cf.data.config.translator.target_language)
    openai_api.initialize(cf.data.config.openai.api_key)
    try:
        storage.get_data()
        print("success.")
        return
    except JsonStorageFormatException as e:
        print("failed.")
        print(f"Failed to parse persistent data file. {e}")
        sys.exit(1)
    except JsonStorageFileException as e:
        pass
    except Exception as e:
        print("failed.")
        print(f"Failed to load persistent data file. {e}")
        sys.exit(1)

    try:
        print("Creating new persistent data file... ", end="")
        sys.stdout.flush()
        storage.set_data([])
        print("success.")
        print("Failed to load persistent data file. Switching to creating a new one.")
    except JsonStorageException as e:
        print("failed.")
        print(f"Failed to create persistent data file. {e}")
        sys.exit(1)


def run_scrape_metadata(novel_origin: str, novel_code: str) -> None:
    setup()
    origin = OriginFactory.get_origin(novel_origin)
    print(f"Scraping metadata for novel: {novel_code}")
    
    try:
        print("(1/3) Loading local storage... ", end="")
        storage = JsonStorage()
        novels = storage.get_data()
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to load local storage. {e}")
        sys.exit(1)

    try:
        print("(2/3) Scraping metadata... ", end="")
        sys.stdout.flush()
        novel_data = origin.process_novel(novel_code)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to scrape metadata. {e}")
        sys.exit(1)

    try:
        print("(3/3) Saving novel data to local storage... ", end="")
        sys.stdout.flush()
        if any(novel_old.novel_code == novel_data.novel_code and novel_old.novel_origin == novel_data.novel_origin for novel_old in novels):
            novel_original = [novel for novel in novels if novel.novel_code == novel_code and novel.novel_origin == novel_origin][0]
            novel_old = copy.deepcopy(novel_original)
            novel_old.title = novel_data.title
            novel_old.author = novel_data.author
            novel_old.description = novel_data.description
            novel_old.author_link = novel_data.author_link if novel_data.author_link else novel_old.author_link
            for chapter in novel_data.chapters:
                if chapter not in novel_old.chapters:
                    novel_old.chapters.append(chapter)
            novel_data = novel_old
            novels.remove(novel_original)
        novels.append(novel_data)
        storage.set_data(novels)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to save novel data to local storage. {e}")
        sys.exit(1)

    print("Done.")

def run_scrape_chapters(novel_origin: str, novel_code: str, chapter_targets_str: str) -> None:
    setup()
    origin = OriginFactory.get_origin(novel_origin)
    print(f"Scraping chapters for novel: {novel_code}")

    try:
        print("(1/4) Parsing targets... ", end="")
        sys.stdout.flush()
        chapter_targets = parse_chapters(chapter_targets_str)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to parse targets. {e}")
        sys.exit(1)

    try:
        print("(2/4) Loading local storage... ", end="")
        storage = JsonStorage()
        novels = storage.get_data()
        novel_old = [novel for novel in novels if novel.novel_code == novel_code and novel.novel_origin == novel_origin][0]
        novel_data = copy.deepcopy(novel_old)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to load local storage. {e}")
        sys.exit(1)

    try:
        print("(3/4) Scraping chapters... ", end="")
        sys.stdout.flush()
        origin.process_targets(novel_data, chapter_targets)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to scrape chapters. {e}")
        sys.exit(1)

    try:
        print(f"(4/4) Saving novel data to local storage... ", end="")
        sys.stdout.flush()
        novels.remove(novel_old)
        novels.append(novel_data)
        storage.set_data(novels)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to save novel data to local storage. {e}")
        sys.exit(1)

    print("Done.")

def run_translate_metadata(novel_origin: str, novel_code: str) -> None:
    setup()
    print(f"Translating metadata for novel: {novel_code}")

    try:
        print(f"(1/4) Loading local storage... ", end="")
        storage = JsonStorage()
        novels = storage.get_data()
        novel_old = [novel for novel in novels if novel.novel_code == novel_code and novel.novel_origin == novel_origin][0]
        novel_data = copy.deepcopy(novel_old)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to load local storage. {e}")
        sys.exit(1)

    try:
        print("(2/4) Initializing translator... ", end="")
        sys.stdout.flush()
        translator = GPTTranslatorSingleton()
        translator.set_original_language(novel_data.original_language)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to initialize translator. {e}")
        sys.exit(1)

    try:
        print(f"(3/4) Translating metadata... ", end="")
        sys.stdout.flush()
        exceptions = translator.translate_novel_metadata(novel_data)
        if exceptions:
            raise Exception(f"Failed to translate metadata.\n\n{exceptions}")
        else:
            print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to translate metadata. {e}")
        sys.exit(1)

    try:
        print(f"(4/4) Saving novel data to local storage... ", end="")
        sys.stdout.flush()
        novels.remove(novel_old)
        novels.append(novel_data)
        storage.set_data(novels)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to save novel data to local storage. {e}")
        sys.exit(1)

    print("Done.")

def run_translate_chapters(novel_origin: str, novel_code: str, chapter_targets_str: str) -> None:
    setup()
    print(f"Translating chapters for novel: {novel_code}")

    try:
        print("(1/13) Parsing targets... ", end="")
        sys.stdout.flush()
        chapter_targets = parse_chapters(chapter_targets_str)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to parse targets. {e}")
        sys.exit(1)

    try:
        print("(2/13) Loading local storage... ", end="")
        storage = JsonStorage()
        novels = storage.get_data()
        novel_old = [novel for novel in novels if novel.novel_code == novel_code and novel.novel_origin == novel_origin][0]
        novel_data = copy.deepcopy(novel_old)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to load local storage. {e}")
        sys.exit(1)

    try:
        print("(3/13) Initializing translator... ", end="")
        sys.stdout.flush()
        translator = GPTTranslatorSingleton()
        translator.set_original_language(novel_data.original_language)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to initialize translator. {e}")
        sys.exit(1)

    try:
        print("(4/13) Generating summaries... ", end="")
        sys.stdout.flush()
        exceptions = translator.summarize_sub_chapters(novel_data, chapter_targets)
        if exceptions:
            raise Exception(f"Failed to generate summaries.\n\n{exceptions}")
        else:
            print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to generate summaries. {e}")
        sys.exit(1)

    try:
        print("(5/13) Saving novel data to local storage... ", end="")
        sys.stdout.flush()
        novels.remove(novel_old)
        novels.append(novel_data)
        storage.set_data(novels)
        novel_old = novel_data
        novel_data = copy.deepcopy(novel_old)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to save novel data to local storage. {e}")
        sys.exit(1)

    try:
        print("(6/13) Updating novel terms sheet... ", end="")
        sys.stdout.flush()
        exceptions = translator.gather_terms_for_sub_chapters(novel_data, chapter_targets)
        if exceptions:
            raise Exception(f"Failed to update novel terms sheet.\n\n{exceptions}")
        else:
            print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to update novel terms sheet. {e}")
        sys.exit(1)

    try:
        print("(7/13) Saving novel data to local storage... ", end="")
        sys.stdout.flush()
        novels.remove(novel_old)
        novels.append(novel_data)
        storage.set_data(novels)
        novel_old = novel_data
        novel_data = copy.deepcopy(novel_old)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to save novel data to local storage. {e}")
        sys.exit(1)

    try:
        print("(8/13) Updating terms sheet weights... ", end="")
        sys.stdout.flush()
        novel_data.terms_sheet.update_dimensions(novel_data.original_body(), novel_data.original_language)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to update terms sheet weights. {e}")
        sys.exit(1)

    try:
        print("(9/13) Saving novel data to local storage... ", end="")
        sys.stdout.flush()
        novels.remove(novel_old)
        novels.append(novel_data)
        storage.set_data(novels)
        novel_old = novel_data
        novel_data = copy.deepcopy(novel_old)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to save novel data to local storage. {e}")
        sys.exit(1)

    try:
        print("(10/13) Translating chapters... ", end="")
        sys.stdout.flush()
        exceptions = translator.translate_sub_chapters(novel_data, chapter_targets)
        if exceptions:
            raise Exception(f"Failed to translate chapters.\n\n{exceptions}")
        else:
            print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to translate chapters. {e}")
        sys.exit(1)

    try:
        print("(11/13) Saving novel data to local storage... ", end="")
        sys.stdout.flush()
        novels.remove(novel_old)
        novels.append(novel_data)
        storage.set_data(novels)
        novel_old = novel_data
        novel_data = copy.deepcopy(novel_old)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to save novel data to local storage. {e}")
        sys.exit(1)

    try:
        print("(12/13) Translating targets' metadata... ", end="")
        sys.stdout.flush()
        exceptions = translator.translate_sub_chapters_metadata(novel_data, chapter_targets)
        if exceptions:
            raise Exception(f"Failed to translate targets' metadata.\n\n{exceptions}")
        else:
            print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to translate targets' metadata. {e}")
        sys.exit(1)

    try:
        print("(13/13) Saving novel data to local storage... ", end="")
        sys.stdout.flush()
        novels.remove(novel_old)
        novels.append(novel_data)
        storage.set_data(novels)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to save novel data to local storage. {e}")
        sys.exit(1)

    print("Done.")

def run_export_chapters(novel_origin: str, novel_code: str, chapter_targets_str: str) -> None:
    setup()
    print(f"Exporting chapters from {novel_code}...")

    try:
        print("(1/3) Parsing targets... ", end="")
        sys.stdout.flush()
        chapter_targets = parse_chapters(chapter_targets_str)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to parse targets. {e}")
        sys.exit(1)

    try:
        print("(2/3) Loading local storage... ", end="")
        sys.stdout.flush()
        storage = JsonStorage()
        novels = storage.get_data()
        novel_old = [novel for novel in novels if novel.novel_code == novel_code and novel.novel_origin == novel_origin][0]
        novel_data = copy.deepcopy(novel_old)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to load local storage. {e}")
        sys.exit(1)

    try:
        print("(3/3) Exporting novel to epub... ", end="")
        sys.stdout.flush()
        cf = Config()
        md_text = write_novel_md(novel_data, chapter_targets)
        output = os.path.join(cf.vars["output_path"], f"{novel_origin}-{novel_data.novel_code}-{cf.data.config.translator.target_language}.epub")
        write_md_as_epub(md_text, output)
        print("success.")
    except Exception as e:
        print("failed.")
        print(f"Failed to export novel to epub. {e}")
        sys.exit(1)

    print("Done.")

