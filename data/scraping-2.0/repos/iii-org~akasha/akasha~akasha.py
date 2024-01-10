import pathlib
import time
from tqdm import tqdm
from typing import Callable, Union, List
from langchain.chains.question_answering import load_qa_chain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import akasha.helper as helper
import akasha.search as search
import akasha.format as format
import akasha.prompts as prompts
import akasha.db
import datetime
from dotenv import load_dotenv

load_dotenv(pathlib.Path().cwd() / ".env")


def aiido_upload(
    exp_name,
    params: dict = {},
    metrics: dict = {},
    table: dict = {},
    path_name: str = "",
):
    """upload params_metrics, table to mlflow server for tracking.

    Args:
        **exp_name (str)**: experiment name on the tracking server, if not found, will create one .\n
        **params (dict, optional)**: parameters dictionary. Defaults to {}.\n
        **metrics (dict, optional)**: metrics dictionary. Defaults to {}.\n
        **table (dict, optional)**: table dictionary, used to compare text context between different runs in the experiment. Defaults to {}.\n
    """
    import aiido

    if "model" not in params or "embeddings" not in params:
        aiido.init(experiment=exp_name, run=path_name)

    else:
        mod = params["model"].split(":")
        emb = params["embeddings"].split(":")[0]
        sea = params["search_type"]
        aiido.init(experiment=exp_name,
                   run=emb + "-" + sea + "-" + "-".join(mod))

    aiido.log_params_and_metrics(params=params, metrics=metrics)

    if len(table) > 0:
        aiido.mlflow.log_table(table, "table.json")
    aiido.mlflow.end_run()
    return


def detect_exploitation(
    texts: str,
    model: str = "openai:gpt-3.5-turbo",
    verbose: bool = False,
    record_exp: str = "",
):
    """check the given texts have harmful or sensitive information

    Args:
        **texts (str)**: texts that we want llm to check.\n
        **model (str, optional)**: llm model name. Defaults to "openai:gpt-3.5-turbo".\n
        **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
        **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set
            record_exp as experiment name.  default ''.\n

    Returns:
        str: response from llm
    """

    logs = []
    model = helper.handle_model(model, logs, verbose)
    sys_b, sys_e = "<<SYS>>\n", "\n<</SYS>>\n\n"
    system_prompt = (
        "[INST]" + sys_b +
        "check if below texts have any of Ethical Concerns, discrimination, hate speech, "
        +
        "illegal information, harmful content, Offensive Language, or encourages users to share or access copyrighted materials"
        + " And return true or false. Texts are: " + sys_e + "[/INST]")

    template = (system_prompt + """ 
    
    Texts: {texts}
    Answer: """)
    prompt = PromptTemplate(template=template, input_variables={"texts"})
    response = LLMChain(prompt=prompt, llm=model).run(texts)
    print(response)
    return response


def openai_vision(pic_path: Union[str, List[str]],
                  prompt: str,
                  model: str = "gpt-4-vision-preview",
                  max_token: int = 3000,
                  verbose: bool = False,
                  record_exp: str = ""):

    start_time = time.time()

    ### process input message ###
    base64_pic = []
    pic_message = []
    if isinstance(pic_path, str):
        pic_path = [pic_path]

    for path in pic_path:
        if not pathlib.Path(path).exists():
            print(f"image path {path} not exist")
        else:
            base64_pic.append(helper.image_to_base64(path))

    for pic in base64_pic:
        pic_message.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{pic}",
                "detail": "auto",
            },
        })
    content = [{"type": "text", "text": prompt}]
    content.extend(pic_message)

    ### call model ###
    import os
    from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
    from langchain.schema.messages import HumanMessage, SystemMessage
    from langchain.callbacks import get_openai_callback

    if ("AZURE_API_TYPE" in os.environ and os.environ["AZURE_API_TYPE"]
            == "azure") or ("OPENAI_API_TYPE" in os.environ
                            and os.environ["OPENAI_API_TYPE"] == "azure"):
        modeln = model.replace(".", "")
        api_base, api_key, api_version = helper._handle_azure_env()
        chat = AzureChatOpenAI(
            deployment_name=modeln,
            temperature=0.0,
            base_url=api_base,
            api_key=api_key,
            api_version=api_version,
            max_tokens=max_token,
        )
    else:
        chat = ChatOpenAI(model=model,
                          max_tokens=max_token,
                          temperature=0.0,
                          verbose=verbose)
    input_message = [HumanMessage(content=content)]

    with get_openai_callback() as cb:
        try:
            ret = chat.invoke(input_message).content
        except Exception as e:
            chat = ChatOpenAI(model="gpt-4-vision-preview",
                              max_tokens=max_token,
                              temperature=0.0)
            ret = chat.invoke(input_message).content

        tokens, prices = cb.total_tokens, cb.total_cost

    end_time = time.time()
    if record_exp != "":
        params = format.handle_params(model, "", "", "", "", "", "ch")
        metrics = format.handle_metrics(0, end_time - start_time, tokens)
        table = format.handle_table(prompt, "\n".join(pic_path), ret)
        aiido_upload(record_exp, params, metrics, table)
    print("\n\n\ncost:", round(prices, 3))

    return ret


class atman:
    """basic class for akasha, implement _set_model, _change_variables, _check_db, add_log and save_logs function."""

    def __init__(
        self,
        chunk_size: int = 1000,
        model: str = "openai:gpt-3.5-turbo",
        verbose: bool = False,
        topK: int = 2,
        threshold: float = 0.2,
        language: str = "ch",
        search_type: Union[str, Callable] = "svm",
        record_exp: str = "",
        system_prompt: str = "",
        max_doc_len: int = 1500,
        temperature: float = 0.0,
    ):
        """initials of atman class

        Args:
            **chunk_size (int, optional)**: chunk size of texts from documents. Defaults to 1000.\n
            **model (str, optional)**: llm model to use. Defaults to "gpt-3.5-turbo".\n
            **verbose (bool, optional)**: show log texts or not. Defaults to False.\n
            **topK (int, optional)**: search top k number of similar documents. Defaults to 2.\n
            **threshold (float, optional)**: the similarity threshold of searching. Defaults to 0.2.\n
            **language (str, optional)**: the language of documents and prompt, use to make sure docs won't exceed
                max token size of llm input.\n
            **search_type (str, optional)**: search type to find similar documents from db, default 'merge'.
                includes 'merge', 'mmr', 'svm', 'tfidf', also, you can custom your own search_type function, as long as your
                function input is (query_embeds:np.array, docs_embeds:list[np.array], k:int, relevancy_threshold:float, log:dict)
                and output is a list [index of selected documents].\n
            **record_exp (str, optional)**: use aiido to save running params and metrics to the remote mlflow or not if record_exp not empty, and set
                record_exp as experiment name.  default "".\n
            **system_prompt (str, optional)**: the system prompt that you assign special instruction to llm model, so will not be used
                in searching relevant documents. Defaults to "".\n
            **max_doc_len (int, optional)**: max document size of llm input. Defaults to 1500.\n
            **temperature (float, optional)**: temperature of llm model from 0.0 to 1.0 . Defaults to 0.0.\n
        """

        self.chunk_size = chunk_size
        self.model = model
        self.verbose = verbose
        self.topK = topK
        self.threshold = threshold
        self.language = language
        self.search_type_str = helper.handle_search_type(
            search_type, self.verbose)
        self.record_exp = record_exp
        self.system_prompt = system_prompt
        self.max_doc_len = max_doc_len
        self.temperature = temperature

        self.timestamp_list = []

    def _set_model(self, **kwargs):
        """change model, embeddings, search_type, temperature if user use **kwargs to change them."""
        ## check if we need to change db, model_obj or embeddings_obj ##
        if "search_type" in kwargs:
            self.search_type_str = helper.handle_search_type(
                kwargs["search_type"], self.verbose)

        if "embeddings" in kwargs:
            self.embeddings_obj = helper.handle_embeddings(
                kwargs["embeddings"], self.verbose)

        if "model" in kwargs or "temperature" in kwargs:
            new_temp = self.temperature
            new_model = self.model
            if "temperature" in kwargs:
                new_temp = kwargs["temperature"]
            if "model" in kwargs:
                new_model = kwargs["model"]
            if new_model != self.model or new_temp != self.temperature:
                self.model_obj = helper.handle_model(new_model, self.verbose,
                                                     new_temp)

    def _change_variables(self, **kwargs):
        """change other arguments if user use **kwargs to change them."""

        ### check input argument is valid or not ###
        for key, value in kwargs.items():
            if (key == "model"
                    or key == "embeddings") and key in self.__dict__:
                self.__dict__[key] = helper.handle_search_type(value)

            elif key in self.__dict__:  # check if variable exist
                if (getattr(self, key, None)
                        != value):  # check if variable value is different
                    self.__dict__[key] = value
            else:
                print(f"argument {key} not exist")

        return

    def _check_db(self):
        """check if user input doc_path is exist or not

        Returns:
            _type_: _description_
        """
        if self.db is None:
            info = "document path not exist or don't have any file.\n"
            print(info)
            return False
        else:
            return True

    def _add_basic_log(self, timestamp: str, fn_type: str):
        """add pre-process log to self.logs

        Args:
            timestamp (str): timestamp of this run
            fn_type (str): function type of this run
        """
        if timestamp not in self.logs:
            self.logs[timestamp] = {}
        self.logs[timestamp]["fn_type"] = fn_type
        self.logs[timestamp]["model"] = self.model
        self.logs[timestamp]["chunk_size"] = self.chunk_size
        self.logs[timestamp]["topK"] = self.topK
        self.logs[timestamp]["threshold"] = self.threshold
        self.logs[timestamp]["language"] = self.language
        self.logs[timestamp]["temperature"] = self.temperature
        self.logs[timestamp]["max_doc_len"] = self.max_doc_len
        self.logs[timestamp]["doc_path"] = self.doc_path

        return

    def _add_result_log(self, timestamp: str, time: float):
        """add post-process log to self.logs

        Args:
            timestamp (str): timestamp of this run
            time (float): spent time of this run
        """
        self.logs[timestamp]["time"] = time
        self.logs[timestamp]["doc_length"] = self.doc_length
        self.logs[timestamp]["doc_tokens"] = self.doc_tokens
        try:
            self.logs[timestamp]["docs"] = "\n\n".join(
                [doc.page_content for doc in self.docs])
            self.logs[timestamp]["doc_metadata"] = "\n\n".join([
                doc.metadata["source"] + "    page: " +
                str(doc.metadata["page"]) for doc in self.docs
            ])
        except:
            self.logs[timestamp]["doc_metadata"] = "none"
            self.logs[timestamp]["docs"] = "\n\n".join(
                [doc for doc in self.docs])

        self.logs[timestamp]["system_prompt"] = self.system_prompt

        return

    def save_logs(self, file_name: str = "", file_type: str = "json"):
        """save logs into json or txt file

        Args:
            file_name (str, optional): file path and the file name. if not assign, use logs/{current time}. Defaults to "".
            file_type (str, optional): the extension of the file, can be txt or json. Defaults to "json".

        Returns:
            plain_text(str): string of the log
        """

        extension = ""
        ## set extension ##
        if file_name != "":
            tst_file = file_name.split(".")[-1]
            if file_type == "json":
                if tst_file != "json" and tst_file != "JSON":
                    extension = ".json"
            else:
                if tst_file != "txt" and tst_file != "TXT":
                    extension = ".txt"

        ## set filename if not given ##
        from pathlib import Path

        if file_name == "":
            file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            logs_path = Path("logs")
            if not logs_path.exists():
                logs_path.mkdir()
            file_path = Path("logs/" + file_name + extension)
        else:
            file_path = Path(file_name + extension)

        ## write file ##
        if file_type == "json":
            import json

            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(self.logs, fp, indent=4, ensure_ascii=False)

        else:
            with open(file_path, "w", encoding="utf-8") as fp:
                for key in self.logs:
                    text = key + ":\n"
                    fp.write(text)
                    for k in self.logs[key]:
                        if type(self.logs[key][k]) == list:
                            text = (k + ": " + "\n".join(
                                [str(w) for w in self.logs[key][k]]) + "\n\n")

                        else:
                            text = k + ": " + str(self.logs[key][k]) + "\n\n"

                        fp.write(text)

                    fp.write("\n\n\n\n")

        print("save logs to " + str(file_path))
        return


class Doc_QA(atman):
    """class for implement search db based on user prompt and generate response from llm model, include get_response and chain_of_thoughts."""

    def __init__(
        self,
        embeddings: str = "openai:text-embedding-ada-002",
        chunk_size: int = 1000,
        model: str = "openai:gpt-3.5-turbo",
        verbose: bool = False,
        topK: int = 2,
        threshold: float = 0.2,
        language: str = "ch",
        search_type: Union[str, Callable] = "svm",
        record_exp: str = "",
        system_prompt: str = "",
        max_doc_len: int = 1500,
        temperature: float = 0.0,
        compression: bool = False,
        use_chroma: bool = False,
        ignore_check: bool = False,
    ):
        super().__init__(
            chunk_size,
            model,
            verbose,
            topK,
            threshold,
            language,
            search_type,
            record_exp,
            system_prompt,
            max_doc_len,
            temperature,
        )
        ### set argruments ###
        self.doc_path = ""
        self.compression = compression
        self.use_chroma = use_chroma
        self.ignore_check = ignore_check
        ### set variables ###
        self.logs = {}
        self.model_obj = helper.handle_model(model, self.verbose,
                                             self.temperature)
        self.embeddings_obj = helper.handle_embeddings(embeddings,
                                                       self.verbose)
        self.embeddings = helper.handle_search_type(embeddings)
        self.model = helper.handle_search_type(model)
        self.search_type = search_type
        self.db = None
        self.docs = []
        self.doc_tokens = 0
        self.doc_length = 0
        self.response = ""
        self.prompt = ""

    def _ask_model(self, system_prompt: str, prompt: str) -> str:

        splitter = '\n----------------\n'

        try:
            text_input = system_prompt + "----------------\n" + splitter.join([doc.page_content for doc in self.docs]) +\
                splitter + prompt

            if self.verbose:
                print("Prompt after formatting:", "\n\n" + text_input)
        except Exception as e:
            print(e)
            print(
                "\n\nText generation encountered an error before querying the language model (LLM).\
                Please check your input data or configuration settings.\n\n")
            response = ""
            return response
        try:
            response = helper.call_model(self.model_obj, text_input)
            response = helper.sim_to_trad(response)

            if response[:8] == "System: ":
                response = response[8:]
        except Exception as e:
            print(e)
            print("\n\nllm error\n\n")

            response = ""
        if self.verbose:
            print("llm response:", "\n\n" + response)

        return response

    def get_response(self, doc_path: Union[List[str], str], prompt: str,
                     **kwargs) -> str:
        """input the documents directory path and question, will first store the documents
        into vectors db (chromadb), then search similar documents based on the prompt question.
        llm model will use these documents to generate the response of the question.

            Args:
                **doc_path (str)**: documents directory path\n
                **prompt (str)**:question you want to ask.\n
                **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
                embeddings, chunk_size, model, verbose, topK, threshold, language , search_type, record_exp,
                system_prompt, max_doc_len, temperature.

            Returns:
                response (str): the response from llm model.
        """
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.doc_path = doc_path
        self.prompt = prompt

        if self.use_chroma:
            self.db, db_path_names = akasha.db.get_db_from_chromadb(
                self.doc_path, self.embeddings)
        else:
            self.db, db_path_names = akasha.db.processMultiDB(
                self.doc_path, self.verbose, self.embeddings_obj,
                self.embeddings, self.chunk_size, self.ignore_check)

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        self.timestamp_list.append(timestamp)
        start_time = time.time()
        if not self._check_db():
            return ""

        self._add_basic_log(timestamp, "get_response")
        self.logs[timestamp]["search_type"] = self.search_type_str
        self.logs[timestamp]["embeddings"] = self.embeddings
        self.logs[timestamp]["compression"] = self.compression

        ### start to get response ###
        self.docs, self.doc_length, self.doc_tokens = search.get_docs(
            self.db,
            self.embeddings_obj,
            self.prompt,
            self.topK,
            self.threshold,
            self.language,
            self.search_type,
            self.verbose,
            self.model_obj,
            self.max_doc_len,
            self.logs[timestamp],
            compression=self.compression,
        )

        if self.docs is None:
            print("\n\nNo Relevant Documents.\n\n")
            return ""

        ## format prompt ##
        if self.system_prompt.replace(' ', '') == "":
            self.system_prompt = prompts.default_doc_ask_prompt()
        prod_sys_prompt, prod_prompt = prompts.format_sys_prompt(
            self.system_prompt, self.prompt)

        self.response = self._ask_model(prod_sys_prompt, prod_prompt)

        end_time = time.time()
        self._add_result_log(timestamp, end_time - start_time)
        self.logs[timestamp]["prompt"] = self.prompt
        self.logs[timestamp]["response"] = self.response
        if self.record_exp != "":
            params = format.handle_params(
                self.model,
                self.embeddings,
                self.chunk_size,
                self.search_type_str,
                self.topK,
                self.threshold,
                self.language,
            )
            metrics = format.handle_metrics(self.doc_length,
                                            end_time - start_time,
                                            self.doc_tokens)
            table = format.handle_table(prompt, self.docs, self.response)
            aiido_upload(self.record_exp, params, metrics, table)

        del self.db
        return self.response

    def chain_of_thought(self, doc_path: Union[List[str], str],
                         prompt_list: list, **kwargs) -> list:
        """input the documents directory path and question, will first store the documents
        into vectors db (chromadb), then search similar documents based on the prompt question.
        llm model will use these documents to generate the response of the question.

        In chain_of_thought function, you can separate your question into multiple small steps so that llm can have better response.
        chain_of_thought function will use all responses from the previous prompts, and combine the documents search from current prompt to generate
        response.

        Args:
           **doc_path (str)**: documents directory path\n
            **prompt (list)**:questions you want to ask.\n
            **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
            embeddings, chunk_size, model, verbose, topK, threshold, language , search_type, record_exp,
            system_prompt, max_doc_len, temperature.

        Returns:
            response (list): the responses from llm model.
        """

        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        if self.system_prompt.replace(' ', '') == "":
            self.system_prompt = prompts.default_doc_ask_prompt()
        self.doc_path = doc_path
        table = {}
        if self.use_chroma:
            self.db, db_path_names = akasha.db.get_db_from_chromadb(
                self.doc_path, self.embeddings)
        else:
            self.db, db_path_names = akasha.db.processMultiDB(
                self.doc_path, self.verbose, self.embeddings_obj,
                self.embeddings, self.chunk_size, self.ignore_check)

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        self.timestamp_list.append(timestamp)
        start_time = time.time()
        if not self._check_db():
            return []

        self._add_basic_log(timestamp, "chain_of_thought")
        self.logs[timestamp]["search_type"] = self.search_type_str
        self.logs[timestamp]["embeddings"] = self.embeddings
        self.logs[timestamp]["compression"] = self.compression

        self.doc_tokens = 0
        self.doc_length = 0
        self.response = []
        self.docs = []
        self.prompt = []
        total_docs = []

        def recursive_get_response(prompt_list):
            pre_result = []
            for prompt in prompt_list:
                if isinstance(prompt, list):
                    response = recursive_get_response(prompt)
                    pre_result.append(Document(page_content="".join(response)))
                else:
                    self.prompt.append(prompt)
                    docs, docs_len, tokens = search.get_docs(
                        self.db,
                        self.embeddings_obj,
                        prompt,
                        self.topK,
                        self.threshold,
                        self.language,
                        self.search_type,
                        self.verbose,
                        self.model_obj,
                        self.max_doc_len,
                        self.logs[timestamp],
                        compression=self.compression,
                    )
                    total_docs.extend(docs)
                    self.doc_length += docs_len
                    self.doc_tokens += tokens
                    self.docs = docs + pre_result
                    ## format prompt ##
                    prod_sys_prompt, prod_prompt = prompts.format_sys_prompt(
                        self.system_prompt, prompt)

                    response = self._ask_model(prod_sys_prompt, prod_prompt)

                    self.response.append(response)
                    pre_result.append(Document(page_content="".join(response)))

                    new_table = format.handle_table(prompt, docs, response)
                    for key in new_table:
                        if key not in table:
                            table[key] = []
                        table[key].append(new_table[key])
            pre_result = []
            return response

        recursive_get_response(prompt_list)
        end_time = time.time()
        self.docs = total_docs
        self._add_result_log(timestamp, end_time - start_time)
        self.logs[timestamp]["prompt"] = self.prompt
        self.logs[timestamp]["response"] = self.response

        if self.record_exp != "":
            params = format.handle_params(
                self.model,
                self.embeddings,
                self.chunk_size,
                self.search_type_str,
                self.topK,
                self.threshold,
                self.language,
            )
            metrics = format.handle_metrics(self.doc_length,
                                            end_time - start_time,
                                            self.doc_tokens)
            aiido_upload(self.record_exp, params, metrics, table)

        del self.db
        return self.response

    def ask_whole_file(self, file_path: str, prompt: str, **kwargs) -> str:
        """input the documents directory path and question, will first store the documents
        into vectors db (chromadb), then search similar documents based on the prompt question.
        llm model will use these documents to generate the response of the question.

            Args:
                **file_path (str)**: document file path\n
                **prompt (str)**:question you want to ask.\n
                **kwargs**: the arguments you set in the initial of the class, you can change it here. Include:\n
                embeddings, chunk_size, model, verbose, topK, threshold, language , search_type, record_exp,
                system_prompt, max_doc_len, temperature.

            Returns:
                response (str): the response from llm model.
        """
        self._set_model(**kwargs)
        self._change_variables(**kwargs)
        self.prompt = prompt
        self.file_path = file_path
        self.docs = akasha.db._load_file(file_path, file_path.split('.')[-1])

        timestamp = datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        self.timestamp_list.append(timestamp)
        start_time = time.time()

        self._add_basic_log(timestamp, "ask_whole_file")
        self.logs[timestamp]["embeddings"] = "ask_whole_file"

        ### start to get response ###
        cur_documents = '\n'.join(
            [db_doc.page_content for db_doc in self.docs])
        self.doc_tokens = self.model_obj.get_num_tokens(cur_documents)

        if self.docs is None:
            print("\n\nNo Relevant Documents.\n\n")
            return ""
        self.doc_length = helper.get_docs_length(self.language, self.docs)

        ## format prompt ##
        if self.system_prompt.replace(' ', '') == "":
            self.system_prompt = prompts.default_doc_ask_prompt()
        prod_sys_prompt, prod_prompt = prompts.format_sys_prompt(
            self.system_prompt, self.prompt)

        self.response = self._ask_model(prod_sys_prompt, prod_prompt)

        end_time = time.time()
        self._add_result_log(timestamp, end_time - start_time)
        self.logs[timestamp]["prompt"] = self.prompt
        self.logs[timestamp]["response"] = self.response
        if self.record_exp != "":
            params = format.handle_params(
                self.model,
                self.embeddings,
                self.chunk_size,
                self.search_type_str,
                self.topK,
                self.threshold,
                self.language,
            )
            metrics = format.handle_metrics(self.doc_length,
                                            end_time - start_time,
                                            self.doc_tokens)
            table = format.handle_table(prompt, self.docs, self.response)
            aiido_upload(self.record_exp, params, metrics, table)

        return self.response
