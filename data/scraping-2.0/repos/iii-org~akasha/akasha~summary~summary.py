from langchain.text_splitter import RecursiveCharacterTextSplitter
import akasha
from pathlib import Path
import time, datetime
import torch, gc
import akasha.db


class Summary(akasha.atman):
    """class for implement summary text file by llm model, include summarize_file method."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 40,
        model: str = "openai:gpt-3.5-turbo",
        verbose: bool = False,
        threshold: float = 0.2,
        language: str = "ch",
        record_exp: str = "",
        format_prompt: str = "",
        system_prompt: str = "",
        max_doc_len: int = 1500,
        temperature: float = 0.0,
    ):
        """initials of Summary class

        Args:
            **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.\n
            **chunk_overlap (int, optional)**: chunk overlap of texts from documents. Defaults to 40.\n
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
            **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
            **threshold (float, optional)**: the similarity threshold of searching. Defaults to 0.2.\n
            **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
                max token size of llm input.\n
            **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set
                record_exp as experiment name.  default "".\n
            **system_prompt (str, optional)**: the system prompt that you assign special instruction to llm model, so will not be used
                in searching relevant documents. Defaults to "".\n
            **max_doc_len (int, optional)**: max doc size of llm document input. Defaults to 3000.\n
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n
        """

        ### set argruments ###
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        self.threshold = threshold
        self.language = language
        self.record_exp = record_exp
        self.format_prompt = format_prompt
        self.system_prompt = system_prompt
        self.max_doc_len = max_doc_len
        self.temperature = temperature

        ### set variables ###
        self.file_name = ""
        self.summary_type = ""
        self.summary_len = 500
        self.logs = {}
        self.model_obj = akasha.helper.handle_model(
            model, self.verbose, self.temperature
        )
        self.model = akasha.helper.handle_search_type(model)
        self.doc_tokens = 0
        self.doc_length = 0
        self.summary = ""
        self.timestamp_list = []

    def _add_log(self, fn_type: str, timestamp: str, time: float, response_list: list):
        """call this method to add log to logs dictionary

        Args:
            fn_type (str): the method current running
            timestamp (str): the method current running timestamp
            time (float): the spent time of the method
            response_list (list): the response list of the method
        """
        if timestamp not in self.logs:
            self.logs[timestamp] = {}
        self.logs[timestamp]["fn_type"] = fn_type
        self.logs[timestamp]["model"] = self.model
        self.logs[timestamp]["chunk_size"] = self.chunk_size

        self.logs[timestamp]["threshold"] = self.threshold
        self.logs[timestamp]["language"] = self.language
        self.logs[timestamp]["temperature"] = self.temperature
        self.logs[timestamp]["max_doc_len"] = self.max_doc_len
        self.logs[timestamp]["file_name"] = self.file_name

        self.logs[timestamp]["time"] = time
        self.logs[timestamp]["doc_length"] = self.doc_length
        self.logs[timestamp]["doc_tokens"] = self.doc_tokens
        self.logs[timestamp]["system_prompt"] = self.system_prompt
        self.logs[timestamp]["format_prompt"] = self.format_prompt
        self.logs[timestamp]["summary_type"] = self.summary_type
        self.logs[timestamp]["summary_len"] = self.summary_len
        self.logs[timestamp]["summaries_list"] = response_list
        self.logs[timestamp]["summary"] = self.summary

    def _set_model(self, **kwargs):
        """change model_obj if "model" or "temperature" changed"""

        if "model" in kwargs or "temperature" in kwargs:
            new_temp = self.temperature
            new_model = self.model
            if "temperature" in kwargs:
                new_temp = kwargs["temperature"]
            if "model" in kwargs:
                new_model = kwargs["model"]
            if new_model != self.model or new_temp != self.temperature:
                self.model_obj = akasha.helper.handle_model(
                    new_model, self.verbose, new_temp
                )

    def _reduce_summary(self, texts: list, tokens: int, total_list: list):
        """Summarize each chunk and merge them until the combined chunks are smaller than the maximum token limit.
        Then, generate the final summary. This method is faster and requires fewer tokens than the refine method.

        Args:
            **texts (list)**: list of texts from documents\n
            **tokens (int)**: used to save total tokens in recursive call.\n
            **total_list (list)**: used to save total response in recursive call.\n

        Returns:
            (list,int): llm response list and total tokens
        """
        response_list = []
        i = 0
        while i < len(texts):
            token, cur_text, newi = akasha.helper._get_text(
                texts, "", i, self.max_doc_len, self.language
            )
            tokens += token

            ### do the final summary if all chunks can be fits into llm model ###
            if i == 0 and newi == len(texts):
                prompt = akasha.prompts.format_reduce_summary_prompt(
                    cur_text, self.summary_len
                )

                response = akasha.helper.call_model(
                    self.model_obj, self.system_prompt + prompt
                )

                total_list.append(response)

                if self.verbose:
                    print("prompt: \n", self.system_prompt + prompt)
                    print("\n\n")
                    print("response: \n", response)
                    print("\n\n\n\n\n\n")

                return total_list, tokens

            prompt = akasha.prompts.format_reduce_summary_prompt(cur_text, 0)

            response = akasha.helper.call_model(
                self.model_obj, self.system_prompt + prompt
            )

            i = newi
            if self.verbose:
                print("prompt: \n", self.system_prompt + prompt)
                print("\n\n")
                print("response: \n", response)
                print("\n\n\n\n\n\n")
            response_list.append(response)
            total_list.append(response)
        return self._reduce_summary(response_list, tokens, total_list)

    def _refine_summary(self, texts: list) -> (list, int):
        """refine summary summarizing a chunk at a time and using the previous summary as a prompt for
        summarizing the next chunk. This approach may be slower and require more tokens, but it results in a higher level of summary consistency.

        Args:
            **texts (list)**: list of texts from documents\n

        Returns:
            (list,int): llm response list and total tokens
        """
        ### setting variables ###
        previous_summary = ""
        i = 0
        tokens = 0
        response_list = []
        ###

        while i < len(texts):
            token, cur_text, i = akasha.helper._get_text(
                texts, previous_summary, i, self.max_doc_len, self.language
            )

            tokens += token
            if previous_summary == "":
                prompt = akasha.prompts.format_reduce_summary_prompt(
                    cur_text, self.summary_len
                )
            else:
                prompt = akasha.prompts.format_refine_summary_prompt(
                    cur_text, previous_summary, self.summary_len
                )

            response = akasha.helper.call_model(
                self.model_obj, self.system_prompt + prompt
            )

            if self.verbose:
                print("prompt: \n", self.system_prompt + prompt)
                print("\n\n")
                print("resposne: \n", response)
                print("\n\n\n\n\n\n")
            response_list.append(response)
            previous_summary = response

        return response_list, tokens

    def summarize_file(
        self,
        file_path: str,
        summary_type: str = "map_reduce",
        summary_len: int = 500,
        output_file_path: str = "",
        **kwargs
    ) -> str:
        """input a file path and return a summary of the file

        Args:
            **file_path (str)**:  the path of file you want to summarize, can be '.txt', '.docx', '.pdf' file.\n
            **summary_type (str, optional)**: summary method, "map_reduce" or "refine". Defaults to "map_reduce".\n
            **summary_len (int, optional)**: _description_. Defaults to 500.\n
            **output_file_path (str, optional)**: the path of output file. Defaults to "".\n
            **kwargs: the arguments you set in the initial of the class, you can change it here. Include:\n
                chunk_size, chunk_overlap, model, verbose, topK, threshold, language , record_exp,
                system_prompt, max_doc_len, temperature.
        Returns:
            str: the summary of the file
        """

        ## set variables ##
        self.file_name = file_path
        self.summary_type = summary_type.lower()
        self.summary_len = summary_len
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        start_time = time.time()
        table = {}
        if not akasha.helper.is_path_exist(file_path):
            print("file path not exist\n\n")
            return ""

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        self.timestamp_list.append(timestamp)

        if self.system_prompt != "" and "<<SYS>>" not in self.system_prompt:
            self.system_prompt = "<<SYS>>" + self.system_prompt + "<<SYS>>"

        # Split the documents into sentences
        documents = akasha.db._load_file(self.file_name, self.file_name.split(".")[-1])
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", " ", ",", ".", "。", "!"],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        docs = text_splitter.split_documents(documents)
        self.doc_length = akasha.helper.get_docs_length(self.language, docs)
        texts = [doc.page_content for doc in docs]

        if summary_type == "refine":
            response_list, self.doc_tokens = self._refine_summary(texts)

        else:
            response_list, self.doc_tokens = self._reduce_summary(texts, 0, [])

        summaries = response_list[-1]
        p = akasha.prompts.format_refine_summary_prompt("", "", self.summary_len)

        ### write summary to file, and if language is chinese , translate it ###

        if self.format_prompt != "":
            if "<<SYS>>" not in self.format_prompt:
                self.format_prompt = "<<SYS>> " + self.format_prompt + "\n\n<<SYS>>"
            response = akasha.helper.call_model(
                self.model_obj,
                self.system_prompt + self.format_prompt + ": \n\n" + summaries,
            )

        elif self.language == "ch":
            response = akasha.helper.call_model(
                self.model_obj,
                "<<SYS>> translate the following text into chinese <<SYS>>: \n\n"
                + summaries,
            )

        self.summary = akasha.helper.sim_to_trad(response)
        self.summary = self.summary.replace("。", "。\n\n")
        ### write summary to file ###
        if output_file_path == "":
            sum_path = Path("summarization/")
            if not sum_path.exists():
                sum_path.mkdir()

            output_file_path = (
                "summarization/" + file_path.split("/")[-1].split(".")[-2] + ".txt"
            )
        elif output_file_path[-4:] != ".txt":
            output_file_path = output_file_path + ".txt"

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(self.summary)

        print(self.summary, "\n\n\n\n")

        end_time = time.time()
        self._add_log("summarize_file", timestamp, end_time - start_time, response_list)
        if self.record_exp != "":
            params = akasha.format.handle_params(
                self.model, "", self.chunk_size, "", -1, -1.0, self.language, False
            )
            params["chunk_overlap"] = self.chunk_overlap
            params["summary_type"] = (
                "refine" if summary_type == "refine" else "map_reduce"
            )
            metrics = akasha.format.handle_metrics(
                self.doc_length, end_time - start_time, self.doc_tokens
            )
            table = akasha.format.handle_table(p, response_list, self.summary)
            akasha.aiido_upload(
                self.record_exp, params, metrics, table, output_file_path
            )
        print("summarization saved in ", output_file_path, "\n\n")

        return self.summary
