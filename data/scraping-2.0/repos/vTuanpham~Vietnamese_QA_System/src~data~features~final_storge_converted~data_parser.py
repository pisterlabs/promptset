import json
import logging
import os
import random
import re
import sys
import string
import multiprocessing
import warnings

sys.path.insert(0, r'./')
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
from httpcore._exceptions import ConnectTimeout
from typing import List, Dict, Union
from abc import ABCMeta, abstractmethod
from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor

from googletrans import Translator

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.data.configs import AdvanceQAExample, AdvanceInstructSample
from src.utils import force_super_call, ForceBaseCallMeta, timeit
from src.data.features.filters import have_code


class DataParser(metaclass=ForceBaseCallMeta):
    def __init__(self, file_path: str,
                 output_dir: str,
                 parser_type: str,
                 max_ctx_wikiset: int = 50000,
                 do_translate: bool = False,
                 do_ctx_augmentation: bool = False,
                 batch_size: int = 12,
                 translate_via: str = 'ggapi',
                 target_fields: List[str] = ['question_text', 'orig_answer_texts'],
                 target_config: Union[AdvanceQAExample, AdvanceInstructSample] = AdvanceInstructSample,
                 max_example_per_thread: int = 400,
                 large_chunks_threshold: int = 20000,
                 no_translated_code: bool = False) -> None:
        self.data_read = None
        self.converted_data = None
        self.file_path = file_path
        self.output_dir = output_dir
        assert os.path.isdir(self.output_dir), "Please provide the correct output directory"

        self.parser_type = parser_type
        self.target_config = target_config

        self.do_translate = do_translate
        self.do_ctx_augmentation = do_ctx_augmentation

        if self.do_ctx_augmentation:
            self.ctx_wiki_dataset = load_dataset("EddieChen372/vietnamese-wiki-segmented",
                                                 split="train")[:max_ctx_wikiset]

        if self.do_translate:
            self.target_fields = target_fields
            self.no_translated_code = no_translated_code
            assert max_example_per_thread < large_chunks_threshold, " Large chunks threshold can't be smaller than max_example per thread!"
            self.max_example_per_thread = max_example_per_thread
            self.large_chunks_threshold = large_chunks_threshold

            self.converted_data_translated = None

            if translate_via != 'ggapi':
                double_quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_type=torch.bfloat16,
                    bnb_4bit_quant_type='nf4'
                )
                self.batch_size = batch_size
                self.tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi",
                                                                     src_lang="en_XX", use_fast=True)
                self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi",
                                                                         device_map='auto',
                                                                         quantization_config=double_quant_config,
                                                                         torch_dtype=torch.bfloat16
                                                                         ).eval()
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.translator = None
            else:
                self.translator = Translator()

    @staticmethod
    def validate(keys: List[str], dataclass: Union[AdvanceQAExample, AdvanceInstructSample] = AdvanceQAExample) -> bool:
        dict_fields = dataclass.get_keys()
        for key in dict_fields:
            assert key in keys, f"\n Invalid parser, the key '{key}' is missing from {dict_fields}\n" \
                                f"you can adjust the fields in the 'src/data/configs/'" \
                                f"  or fill in the missing field"
        return True

    def post_translate_validate(self) -> None:
        validated_translate_data = []
        # Note: This validates will override the original self.converted_data
        for idx, example in enumerate(tqdm(self.converted_data, desc="Validating data for translation:")):
            for key in self.target_fields:
                if self.no_translated_code:
                    example_filters = 0
                    contain_code, score, found_elements = have_code(example[key])
                    if contain_code:
                        example_filters += 1
                        if len(self.converted_data) - 1 == idx:
                            tqdm.write(f"Number of example with code: {example_filters}")
                        break
                    elif key == self.target_fields[-1]:
                        validated_translate_data.append(example)
                else:
                    if key == self.target_fields[-1]: validated_translate_data.append(example)

        print(f"\nTotal data left after filtering for translation: {len(validated_translate_data)}\n")
        self.converted_data = validated_translate_data

    @staticmethod
    def id_generator(size=6, chars=string.ascii_uppercase + string.digits) -> str:
        return ''.join(random.choice(chars) for _ in range(size))

    def inject_random_ctx(self, docs: List[str], max_docs: int = 9, random_range: int = 20) -> List[str]:
        assert self.do_ctx_augmentation, "Please enable context augmentation via self.do_ctx_augmentation"
        assert not self.do_translate, "Please inject random ctx after translation as the dataset for random ctxs " \
                                      "is already in vietnamese."

        if len(docs) == max_docs:
            return docs

        def rm_underscore(data: str) -> str:
            return re.sub('_', " ", data)

        max_dataset_len = len(self.ctx_wiki_dataset)
        idx = random.randint(0, abs(max_dataset_len - random_range))
        random_docs_num = random.randint(1, abs(max_docs - len(docs)))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=len(docs[0]),
            chunk_overlap=len(docs[0]) * 0.3,
            length_function=len,
            add_start_index=False,
            separators=["\n\n", "\n", " ", "", ".", ","],
            keep_separator=True
        )
        texts = text_splitter.create_documents(self.ctx_wiki_dataset['segmented_text'][idx:idx + random_range])
        texts = random.choices(texts, k=random_docs_num)
        random_docs = [rm_underscore(text.page_content) for text in texts]

        random_pos = random.randint(0, len(random_docs))
        final_random_docs_ctx = random_docs[:random_pos] + docs + random_docs[random_pos:]

        return final_random_docs_ctx

    def translate_en2vi_advance_qa(self, example: Dict, translator: Translator = None) -> Dict:
        assert self.do_translate, "Please enable translate via self.do_translate"
        keys = AdvanceQAExample.get_keys()
        for key in keys:
            if key in self.target_fields:
                type = "str" if isinstance(example[key], str) else "list"
                if example[key] == "":
                    continue
                if len(example[key]) > 15000:
                    warnings.warn("\n Example" + example["qas_id"] + " have field len larger than 15000")
                    example[key] = example[key][:15000]
                example[key] = self.translate_en2vi(example[key], type, translator)

        return example

    def translate_en2vi(self, en_texts: Union[List[str], str], data_type: str, translator: Translator = None) -> Union[
        List[str], str]:
        assert self.do_translate, "Please enable translate via self.do_translate"
        if not self.translator:
            if len(en_texts) > self.batch_size and data_type != 'str':
                translated_en_texts = []
                en_texts_batch = [en_texts[x:x + self.batch_size] for x in range(0, len(en_texts), self.batch_size)]
                for batch in en_texts_batch:
                    translated_en_texts += self.translate_en2vi(batch, type(batch))
                return translated_en_texts

            input_ids = self.tokenizer_en2vi(en_texts, padding=True,
                                             return_tensors="pt").to(self.device)
            output_ids = self.model_en2vi.generate(
                **input_ids,
                decoder_start_token_id=self.tokenizer_en2vi.lang_code_to_id["vi_VN"],
                num_return_sequences=1,
                num_beams=5,
                early_stopping=True,
                max_new_tokens=1024
            )
            vi_texts = self.tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
        else:
            translator_instance = self.translator if not translator else translator
            vi_texts = translator_instance.translate(en_texts, src='en', dest='vi')
            vi_texts = [text.text for text in vi_texts] if data_type != 'str' else vi_texts.text
            return vi_texts
        return vi_texts[0] if data_type == 'str' else vi_texts

    @timeit
    def translate_converted(self, en_data: List[str] = None,
                            desc: str = None,
                            translator: Translator = None,
                            large_chunk: List[str] = None) -> Union[None, List[str]]:
        """This function support translation in multithread for large dataset"""

        assert self.converted_data is not None or en_data is not None or large_chunk is not None, \
            "Please implement the convert function for DataParser " \
            "and assign converted_data to self.converted_data"

        if not en_data and not large_chunk:
            converted_data = self.converted_data
        elif not en_data:
            converted_data = large_chunk
        else:
            converted_data = en_data

        translated_data = []

        # Split large data into large chunks, recursive feed to the same function
        if len(converted_data) > self.large_chunks_threshold and large_chunk is None:
            num_large_chunks = len(converted_data) / self.large_chunks_threshold
            large_chunks = [converted_data[x:x + self.large_chunks_threshold] for x in
                            range(0, len(converted_data), self.large_chunks_threshold)]
            tqdm.write(f" Data is way too large, spliting data into {num_large_chunks} large chunk for sequential translation")

            for idx, large_chunk in enumerate(tqdm(large_chunks, desc=f"Translating large chunk ")):
                tqdm.write(f" Processing large chunk No: {idx}")
                self.translate_converted(large_chunk=large_chunk)
            return None

        # Split large chunk into large example, recursive feed to the same function via multithread
        if len(converted_data) > self.max_example_per_thread and en_data is None:
            num_threads = len(converted_data) / self.max_example_per_thread
            chunks = [converted_data[x:x + self.max_example_per_thread] for x in
                      range(0, len(converted_data), self.max_example_per_thread)]
            tqdm.write(f" Data too large, splitting data into {num_threads} chunk, each chunk is {len(chunks[0])}"
                       f" Processing with multithread...")
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                finished_task = 0
                manager = multiprocessing.Manager()

                def callback_done(future):
                    nonlocal translated_data
                    nonlocal finished_task
                    nonlocal manager
                    lock = manager.Lock()
                    if not future.exception():
                        with lock:
                            translated_data += future.result()
                            finished_task += 1
                            tqdm.write("Task finished, adding translated data to result")
                    else:
                        tqdm.write(f"Task failed with the following error {future.exception()}\n"
                                   f" \nrestarting thread when others finished")
                        pass

                for idx, chunk in enumerate(chunks):
                    future_chunk = executor.submit(self.translate_converted, chunk, f"chunk {idx}", Translator())
                    future_chunk.add_done_callback(callback_done)
                    future_dict = {
                        "future": future_chunk,
                        "idx": idx}
                    futures.append(future_dict)

                # Progress bar
                desc = "Translating total converted large chunk data" if large_chunk else "Translating total converted data"
                progress_bar = tqdm(range(finished_task, len(futures)), desc=desc)
                # Manually refresh the progress bar to display it
                progress_bar.refresh()

                # Wait for all threads to complete
                while finished_task < len(futures):
                    progress_bar.refresh()
                    for future_dict in futures:
                        # If exception occurs in one of the thread, restart the thread with its specific chunk
                        if future_dict['future'].exception():
                            tqdm.write(
                                f" Thread {future_dict['idx']} failed, restarting thread with chunk {future_dict['idx']}")
                            backup_future_chunk = executor.submit(self.translate_converted, chunks[future_dict['idx']],
                                                                  f"Backup chunk {future_dict['idx']}", Translator())
                            backup_future_chunk.add_done_callback(callback_done)
                            backup_future_dict = {"future": backup_future_chunk,
                                                  "idx": future_dict['idx']}
                            futures[future_dict['idx']] = backup_future_dict
                            continue
                        elif future_dict['future'].result():
                            continue
                        # translated_data += future_dict['future'].result()

            if large_chunk:
                if not self.converted_data_translated:
                    self.converted_data_translated = translated_data
                else:
                    self.converted_data_translated += translated_data
                return None

            self.converted_data_translated = translated_data
            return None

        try:
            progress_bar_desc = "Translating converted data" if not desc else f"Translating converted data {desc}"
            for example in tqdm(converted_data, desc=progress_bar_desc):
                translated_data_example = self.translate_en2vi_advance_qa(example, translator)
                translated_data.append(translated_data_example)
            if en_data: return translated_data
            self.converted_data_translated = translated_data
        except ConnectTimeout:
            if not desc:
                raise f" Connection timeout, please provide better connection"
            else:
                tqdm.write(f" Connection timeout from thread {desc}")
                raise f" Connection timeout raise from thread {desc}"

    @abstractmethod
    @force_super_call
    @timeit
    def convert(self) -> Union[List[Dict], None]:
        assert self.data_read is not None, "Please implement the read function for DataParser" \
                                           " and assign data to self.data_read"
        pass

    @abstractmethod
    @force_super_call
    @timeit
    def read(self) -> Union[List, Dict, None]:
        assert os.path.isfile(self.file_path), f"Invalid path file for {self.file_path}"
        pass

    @property
    @force_super_call
    @timeit
    def save(self) -> None:
        output_path = os.path.join(self.output_dir, f"{self.parser_type}.json")
        with open(output_path, 'w', encoding='utf-8') as jfile:
            print(f"\n Saving {self.parser_type} to {output_path}... ")
            validated_data = []
            for idx, data in enumerate(tqdm(self.converted_data, desc="Writing data to file")):
                if self.validate(self.converted_data[idx].keys(), self.target_config):
                    # validated_data.append(data)
                    jfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            # json.dump(validated_data, jfile, ensure_ascii=False, indent=4)
            print(f"\n Total line printed: {idx + 1}")

        if IN_COLAB:
            print(f"\n Downloading converted data to local machine...")
            files.download(output_path)

        if self.do_translate:
            self.post_translate_validate()
            self.translate_converted()
            assert self.converted_data_translated is not None, "Converted data haven't been translated yet!"
            output_translated_path = os.path.join(self.output_dir, f"{self.parser_type}_translated.json")
            with open(output_translated_path, 'w', encoding='utf-8') as jfile:
                print(f"\n Saving {self.parser_type} translated to {output_translated_path}... ")
                translated_data = []
                for idx, data in enumerate(tqdm(self.converted_data_translated, desc="Writing translated data to file")):
                    # translated_data.append(data)
                    jfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                # json.dump(translated_data, jfile, ensure_ascii=False, indent=4)
                print(f"\n Total line printed: {idx + 1}")

            if IN_COLAB:
                print(f"\n Downloading converted translated data to local machine...")
                files.download(output_translated_path)

