# llm_chunker is a class in charge of chunking a large document > 30k tokens in multiple chunks
# Each chunk is a "smart" chunk prepared for a Retrieval Augmented Generation purpose
# To perform a smart chunking, llm_chunker takes as parameters:
# - expected_task: the task for which the chunker is used (e.g. "summarization", "translation", "question_generation")
# - max_tokens: the maximum number of tokens allowed in a chunk
# - document_type: a description of the document type (e.g. "news", "wikipedia", "scientific_paper", "rfc_specification")

# The chunker is based on the following principles:
# - the chunker runs a first pass to generate an instruction for chunking
# - the instructions are given to a self reflection task to finally prepare a smart chunking strategy
# - the llm_chunker performs the smart chunking and returns a list of chunks

import os
import openai
import jinja2

import tiktoken


class llm_chunker:
    def __init__(self, expected_task, max_tokens, document_type, openai_api_key):
        self.expected_task = expected_task
        self.max_tokens = max_tokens
        self.num_lines_per_section = 100
        self.document_type = document_type
        self.openai_api_key = openai_api_key
        self.chunker = None
        self.chunker_path = None
        self.chunker_name = None
        self.chunker_version = None
        self.chunker_config = None
        self.document = None
        self.total_tokens = None

        self.first_pass_instructions = []

    def load_prompt(self, prompt_name):
        # load the prompt from the prompt library the prompt library is a text file with name as "prompt_name.txt" in the "prompts" folder

        prompt = open("prompts/" + prompt_name + ".txt", "r")
        prompt = prompt.read()
        return prompt

    def load_document(self, document_file_path):
        self.document = document_file_path

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def call_chat_openai(self, system_prompt, user_prompt):
        openai.api_key = os.getenv("OPENAI_API_KEY")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[

                  {
                      "role": "system",
                      "content": system_prompt
                  },
                {
                      "role": "user",
                      "content": user_prompt
                  }

            ],
            temperature=0,
            max_tokens=544,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message

    # this method runs a fist pass on the entire document, given the document type and description
    # to prepare a relevant instruction for the next llm chunk task
    def prepare_chunking_instruction(self, section_start, section_end):

        # load the prompts for the first pass
        system_prompt = self.load_prompt("first_pass_system_message")

        # template prompt complete with the "total_tokens" parameter
        template_system_prompt = jinja2.Template(system_prompt)
        system_prompt = template_system_prompt.render(total_tokens=self.total_tokens, document_type=self.document_type, expected_task=self.expected_task)

        user_prompt = self.load_prompt("first_pass_user_message")

        # template user prompt complete with the "document_section", "total_tokens", "section_end", "section_start" parameters
        template_user_prompt = jinja2.Template(user_prompt)
        user_prompt = template_user_prompt.render(document_section=self.document, total_tokens=self.total_tokens, section_end=section_end, section_start=section_start , expected_task=self.expected_task)

        # call the chat API
        response = self.call_chat_openai(system_prompt, user_prompt)

        # extract the instruction from the response
        instruction = response

        self.first_pass_instructions.append(instruction)

    # this method performs the first pass on the entire document: split the document in naive sections

    def first_pass(self):

        # split the document in naive sections
        sections = []

        with open(self.document, 'r') as file:
            #self.total_tokens = self.num_tokens_from_string(" ".join(file), "cl100k_base")
            section = []
            for line in file:
                section.append(line.strip())
                if len(section) == self.num_lines_per_section:
                    sections.append(section)
                    section = []
            if len(section) > 0:
                sections.append(section)


        # for each naive section, prepare a chunking instruction
        token_position = 0
        for naive_section in sections:
            # convert the naive section to a string
            section_length = self.num_tokens_from_string(
                " ".join(naive_section), "cl100k_base")
            self.prepare_chunking_instruction(
                token_position, token_position + section_length)
            token_position = token_position + section_length


# main program to run a test from the file
if __name__ == "__main__":

    auto_chunker = llm_chunker(
        expected_task="question_and_answers",
        max_tokens=1000,
        document_type="rfc_specification",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    rfc_file = 'ietf.org_rfc_rfc9340.txt'

    auto_chunker.load_document(rfc_file)
    auto_chunker.first_pass()
