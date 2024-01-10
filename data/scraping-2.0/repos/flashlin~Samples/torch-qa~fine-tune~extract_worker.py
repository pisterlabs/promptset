import os
from io_utils import query_sub_files
from langchain_lit import load_markdown_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
from qa_file_utils import is_match, ANSWER_PATTERN, convert_qa_md_file_to_train_jsonl
from llama_cpp import Llama
import time
import re
from finetune_utils import load_finetune_config
from finetune_lit import load_peft_model, ask_llama2_instruction_prompt, get_finetune_model_name
from qa_file_utils import query_qa_file
import shutil

MODEL_NAME = "Mistral-7B-Instruct-v0.1"

def get_args():
    parser = argparse.ArgumentParser(description="Extract Text ")
    parser.add_argument("model_name", nargs='?', help="Name of your model")
    args = parser.parse_args()
    return args

def load_gguf_model(model_name):
   llm = Llama(
      model_path=f"../models/{model_name}.gguf",
      n_ctx=8192,
      n_threads=12,
      n_gpu_layers=128,
      verbose=False
   )
   return llm

DEFAULT_PROMPT_TEMPLATE = """<s>[INST] {user_input} [/INST]"""
PROMPT_TEMPLATES = {
   "wizardlm-30b": "{user_input}\n\n### Response:",
}
def get_prompt_template(model_name):
   for model_pattern, prompt_template in PROMPT_TEMPLATES.items():
      pattern = re.compile(model_pattern, re.IGNORECASE)
      match = pattern.match(model_name)
      if match:
         return prompt_template
   return DEFAULT_PROMPT_TEMPLATE

PROMPT = get_prompt_template(MODEL_NAME)
print(f"{PROMPT=}")

def get_prompt(sys_instruction, user_input):
   global PROMPT
   prompt = PROMPT
   return prompt.format(sys_instruction=sys_instruction, user_input=user_input)


def ask(query):
   global llm
   prompt = get_prompt(None, query)
   output = llm(
      #"### Human: {prompt}\n\n### Assistant:",
      prompt,
      max_tokens=512,  # Generate up to 512 tokens
      stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
      echo=False        # Whether to echo the prompt
   )
   # return output
   resp = output['choices'][0]
   answer = resp['text']
   return answer


class ParagraphsContext:
    def __init__(self):
        self.read_state = ParagraphReadyState(self)
        self.paragraph_list = []
        self.yield_fn = None

    def read_line(self, line: str):
        self.read_state.read_line(line)

    def output_paragraph(self, paragraph: str):
        self.paragraph_list.append(paragraph)
        if self.yield_fn is not None:
            self.yield_fn(paragraph)

    def flush(self):
        self.read_state.flush()


class ParagraphReadyState:
    def __init__(self, context: ParagraphsContext):
        self.context = context
        self.buff = ""

    def read_line(self, line: str):
        if line.strip() == "":
            self.flush()
            return
        if self.buff != "":
            self.buff += "\n"
        answer_captured_text = is_match(line, ANSWER_PATTERN)
        if answer_captured_text is not None:
            self.buff += ' '
        self.buff += line

    def flush(self):
        if self.buff == "":
            return
        self.context.paragraph_list.append(self.buff)
        self.buff = ""


def extract_text(folder: str):
    files = query_sub_files(folder, ['.txt', '.md'])
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            yield file, f.read()


def split_text_to_paragraphs(text: str):
    paragraph_parser = ParagraphsContext()
    for line in text.split('\n'):
        paragraph_parser.read_line(line)
    paragraph_parser.flush()
    for paragraph in paragraph_parser.paragraph_list:
        yield paragraph


def extract_paragraphs(folder: str):
    for source, text in extract_text(folder):
        for paragraph in split_text_to_paragraphs(text):
            yield source, paragraph


def first_element(input_iterable):
    iterator = iter(input_iterable)
    return next(iterator)


#################
def generate_question(llm, content: str):
   with open('./prompt.txt', 'r', encoding='utf-8') as f:
       template = f.read()
   prompt = template.format(content=content)
   answer = llm.ask(prompt)
   return answer


def extrac_question_body(question_line: str):
    match = re.match(f'\d+. (.*)', question_line)
    if match:
        return match.group(1).strip()
    return None


def split_questions_content(content: str):
    question_lines = re.findall(r'\d+\. .*', content)
    questions = []
    for question_line in question_lines:
        q = extrac_question_body(question_line)
        questions.append(q)
    return questions


class LLM:
    def __init__(self):
        model, tokenizer, generation_config = self.load_model()
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.device = 'cuda'

    def load_model(self):
        base_model = f"../models/{MODEL_NAME}"
        model, tokenizer = load_peft_model(base_model, None)
        generation_config = model.generation_config
        generation_config.max_new_tokens = 1024
        generation_config.temperature = 0.2
        generation_config.do_sample = True
        generation_config.top_p = 0.9
        generation_config.num_return_sequences = 1
        generation_config.pad_token_id = tokenizer.eos_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        return model, tokenizer, generation_config

    def ask(self, user_input: str):
        answer = ask_llama2_instruction_prompt(model=self.model,
                                               generation_config=self.generation_config,
                                               tokenizer=self.tokenizer,
                                               device=self.device,
                                               question=user_input)
        return answer


def append_qa_data(content: str, llm_qa_file: str):
    with open(llm_qa_file, 'a', encoding='utf-8') as f:
        f.write(content)
        f.write("\n\n\n")


def read_file(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()


def generate_qa_data(llm, content: str, output_llm_qa_file):
    answer = generate_question(llm, content)
    answer = answer.strip()
    append_qa_data(answer, output_llm_qa_file)
    print(f"{answer}")
    print("-------------------------------------------------------------------")
    print("\n\n\n")


if __name__ == '__main__':
    args = get_args()
    # folder = './data-user'
    # with open('./results/paragraphs.txt', 'w', encoding='utf-8') as f:
    #     for source, paragraph in extract_paragraphs(folder):
    #         f.write("Question: ???\r\n")
    #         f.write(f"Answer: {paragraph}\r\n")
    #         f.write('\r\n\r\n')
    #
    # source, first_paragraph = first_element(extract_paragraphs(folder))
    # print(f"{first_paragraph=}")

    start_time = time.time()
    llm = LLM()
    # llm = load_gguf_model(MODEL_NAME)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{MODEL_NAME} Loaded. Take {execution_time} sec.")

    output_llm_qa_file = f"./results/llm-qa.md"
    if os.path.exists(output_llm_qa_file):
        os.remove(output_llm_qa_file)

    # while True:
    #     user_input = input("query: ")
    #     if user_input == '/bye':
    #         break
    #     with open('./content.txt', 'r', encoding='utf-8') as f:
    #         content = f.read()
    #     generate_qa_data(llm, content=content, output_llm_qa_file=output_llm_qa_file)

    qa_jsonl_file = './results/qa.jsonl'
    for idx, file in enumerate(query_sub_files('./data', ['.txt', '.md'])):
        content = read_file(file)
        print(f"process {file}")
        append_qa_data(f'# {file}', output_llm_qa_file)
        generate_qa_data(llm, content=content, output_llm_qa_file=output_llm_qa_file)
        # convert_qa_md_file_to_train_jsonl(file, 'a')
        shutil.move(file, './data-processed/')
